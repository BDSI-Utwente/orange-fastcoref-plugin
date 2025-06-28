from functools import partial
from Orange.data import Table
from Orange.widgets import gui
from concurrent.futures import Future, wait
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from AnyQt.QtCore import Qt, pyqtSlot, QThread
from AnyQt import QtCore
import traceback

from orangecontrib.text.corpus import Corpus              # type: ignore warning from shared namespace
from orangecontrib.text.preprocess import BASE_TOKENIZER  # type: ignore warning from shared namespace

import spacy
import spacy.cli
import spacy.cli.download

from fastcoref import spacy_component  # noqa: F401 imported for side-effects

import pandas as pd


SPACY_MODELS = [
    "en_core_web_sm",
    "en_core_web_md",
    "en_core_web_lg",
    "nl_core_news_sm",
    "nl_core_news_md",
    "nl_core_news_lg",
]
FASTCOREF_MODELS = ["fastcoref", "lingmess"]

DEFAULT_SPACY_MODEL = "nl_core_news_md"
DEFAULT_FASTCOREF_MODEL = "fastcoref"

LABEL_STYLE_COLOURS = {
  "inactive": {
    "fg": "#6B7280",
    "bg": "#F3F4F6"
  },
  "active": {
    "fg": "#2563EB",
    "bg": "#DBEAFE"
  },
  "success": {
    "fg": "#059669",
    "bg": "#D1FAE5"
  },
  "error": {
    "fg": "#DC2626",
    "bg": "#FEE2E2"
  }
}
LABEL_STYLE_TEMPLATE = """ QLabel {{
    padding: 0.2em; margin-bottom: 6px;
    color: {fg}; border: 1px solid {fg}; background-color: {bg};
}} """

LABEL_STYLE_INACTIVE = LABEL_STYLE_TEMPLATE.format(**LABEL_STYLE_COLOURS["inactive"])
LABEL_STYLE_ACTIVE = LABEL_STYLE_TEMPLATE.format(**LABEL_STYLE_COLOURS["active"])
LABEL_STYLE_SUCCESS = LABEL_STYLE_TEMPLATE.format(**LABEL_STYLE_COLOURS["success"])
LABEL_STYLE_ERROR = LABEL_STYLE_TEMPLATE.format(**LABEL_STYLE_COLOURS["error"])


class Task:
    future: Future = None
    watcher: FutureWatcher = None
    cancelled = False
    callback: callable = None

    def cancel(self):
        self.cancelled = True
        if self.future:
            self.future.cancel()
        if self.watcher:
            self.watcher.disconnect()
        wait([self.future])


class CoreferenceResolutionWidget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Coreference Resolution"
    description = "Coreference resolution widget for Orange3 using FastCoRef models."
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "fastcoref", "coreference resolution"]
    want_main_area = False
    resizing_enabled = True

    # data
    corpus: Corpus | None = None  # input corpus
    nlp: spacy.language.Language | None = None  # spaCy NLP object

    # threaded tasks
    _task: Task | None = None  # currently running task
    _executor: ThreadExecutor  # executor for running tasks in a separate thread

    # settings
    commitOnChange = Setting(True)
    coref_model = Setting(DEFAULT_FASTCOREF_MODEL) 
    spacy_model = Setting(DEFAULT_SPACY_MODEL)  

    # 'content' column for input data, depends on domain
    settingsHandler = DomainContextHandler()
    content_attr = ContextSetting("Text")

    class TaskCancelledException(Exception): 
        def __init__(self): 
            super().__init__("task cancelled")

    class Inputs:
        # specify the name of the input and the type
        corpus = Input("Corpus", Corpus)

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        resolved_corpus = Output("Corpus", Corpus, default=True)
        coreferences = Output("Coreferences", Table)

    # same class can be initiated for Error and Information messages
    class Warning(OWWidget.Warning):
        warning = Msg("Something bad happened!")

    class Information(OWWidget.Information):
        spacy_download_model = Msg("{} downloading...")
        spacy_load_model = Msg("{} loading...")

    def __init__(self):
        super().__init__()

        # currently running task
        self._task = None
        self._executor = ThreadExecutor()

        self.corpus = None
        self.nlp = None
        self.spacy_model = DEFAULT_SPACY_MODEL
        self.spacy_model_status: str = "no spacy model loaded"
        self.coref_model = DEFAULT_FASTCOREF_MODEL
        self.coref_model_status: str = "no coref model loaded"

        self.controlArea.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.controlArea.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        
        # group container for model-related settings
        self.groupbox_model_settings = gui.vBox(self.controlArea, box="Model settings")
        self.groupbox_model_settings.setMinimumSize(QtCore.QSize(250, 0))
        self.groupbox_model_settings.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

        # row for spacy model selection and status
        # spacy_row: QtWidgets.QGroupBox
        # spacy_row = gui.hBox(self.groupbox_model_settings, spacing = 12)
        
        gui.comboBox(
            self.groupbox_model_settings,
            self,
            "spacy_model",
            items=SPACY_MODELS,
            label="spaCy Model",
            callback=self.spacy_model_changed,
            sendSelectedValue=True,
        )
        self.spacy_model_status_label = gui.label(self.groupbox_model_settings, self, "%(spacy_model_status)s")
        self.spacy_model_status_label.setStyleSheet(LABEL_STYLE_INACTIVE)

        # row for coref model and status
        # coref_row = gui.hBox(self.groupbox_model_settings, spacing = 12)
        gui.comboBox(
            self.groupbox_model_settings,
            self,
            "coref_model",
            items=FASTCOREF_MODELS,
            label="Coreference Model",
            callback=self.coref_model_changed,
            sendSelectedValue=True,
        )

        self.coref_model_status_label = gui.label(self.groupbox_model_settings, self, "%(coref_model_status)s")
        self.coref_model_status_label.setStyleSheet(LABEL_STYLE_INACTIVE)

        # self.coref_model_status_label = gui.label(self.groupbox_model_settings, self, "")
        # self.coref_model_status_label.setVisible(False)
        
        # group container for domain-related settings
        self.groupbox_domain_settings = gui.vBox(self.controlArea, box="Domain settings")
        self.groupbox_domain_settings.setMinimumSize(QtCore.QSize(250, 0))
        self.groupbox_domain_settings.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

        gui.comboBox(self.groupbox_domain_settings, self, "content_attr", label="Content attribute name", sendSelectedValue=True)

        gui.checkBox(
            self.groupbox_domain_settings,
            self,
            "commitOnChange",
            "Auto run on changes",
            callback=self.setting_changed,
        )

        self.start_button = gui.button(self.groupbox_domain_settings, self, "Start", self._start_coref_resolution)
        self.start_button.setDisabled(self.commitOnChange)

        self.stop_button = gui.button(self.groupbox_domain_settings, self, "Stop", self._stop_coref_resolution)
        self.stop_button.setVisible(False)
        

    @Inputs.corpus
    def set_corpus(self, corpus: Corpus):
        self.closeContext()
        
        if corpus:
            self.corpus = corpus

            # clear current items
            self.controls.content_attr.clear()
            self.controls.content_attr.addItems([meta.name for meta in self.corpus.domain.metas])

            # apply context (now that correct options are available, that should hopefully update the selected element)
            self.openContext(corpus.domain)
                                 
        else:
            self.corpus = None

    def setting_changed(self):
        self.start_button.setDisabled(self.commitOnChange)
        self.update_if_needed()

    def update_if_needed(self): 
        if self.commitOnChange:
            self._start_coref_resolution()

    def spacy_model_changed(self):
        self.coref_model_status = "waiting for spacy model..."
        self.coref_model_status_label.setStyleSheet(LABEL_STYLE_INACTIVE)
        self._start_spacy_load_task(self.spacy_model)

    def cancel_or_await_previous_task(self):
        """Cancel the current task if it is running, or wait for it to finish."""
        if self._task is not None:
            self._task.cancel()
            
            # the done signal may or may not be connected to anything. 
            # NOTE: I'm also not convinced this is needed considering we already call 
            # _task.watcher.disconnect() in _task.cancel(). For now, keep the disconnect
            # call and ignore errors raised for lack of connections.
            try:
                self._task.watcher.done.disconnect()
            except TypeError as err: 
                print(f"failed to disconnect signals: {err}.")

            assert self._task.future.done()
            self._task = None
        
        # cleanup any lingering status messages/progress bars
        self.clear_messages()
        self.progressBarFinished()

    def _assert_task_finished(self, future):
        """Assert that the current task has finished."""
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is future
        assert future.done()

    def _start_spacy_load_task(self, model_name):
        """Start a task to load the spaCy model."""
        self.cancel_or_await_previous_task()

        if not spacy.util.is_package(self.spacy_model):
            self._start_spacy_download_task(self.spacy_model)
            return

        self.Information.spacy_load_model(model_name)
        self.spacy_model_status = f"loading model: {model_name}..."
        self.spacy_model_status_label.setStyleSheet(LABEL_STYLE_ACTIVE)
        self.setInvalidated(True)
        self.setReady(False)

        self._task = task = Task()
        task.future = self._executor.submit(spacy.load, model_name)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._spacy_load_task_finished)

    @pyqtSlot(Future)
    def _spacy_load_task_finished(self, future):
        # assert that we're looking at the correct task, and that it has finished
        self._assert_task_finished(future)

        # clear task, info message
        self._task = None
        self.Information.spacy_load_model.clear()
        self.spacy_model_status = f"{self.spacy_model} ready"
        self.spacy_model_status_label.setStyleSheet(LABEL_STYLE_SUCCESS)

        # load the spaCy model
        self.nlp = future.result()

        # start loading the coreference model
        self._start_coref_load_task(self.coref_model)

    def _start_spacy_download_task(self, model_name):
        self.cancel_or_await_previous_task()

        self.Information.spacy_download_model(model_name)
        self.spacy_model_status = f"downloading model: {model_name}..."
        self.spacy_model_status_label.setStyleSheet(LABEL_STYLE_ACTIVE)
        self.setInvalidated(True)
        self.setReady(False)

        self._task = task = Task()
        task.future = self._executor.submit(spacy.cli.download, model_name)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._spacy_download_task_finished)

    @pyqtSlot(Future)
    def _spacy_download_task_finished(self, future):
        self._assert_task_finished(future)

        # clear task, info message
        self._task = None
        self.Information.spacy_download_model.clear()
        self.spacy_model_status = f"downloading model: {self.spacy_model} completed"
        self.spacy_model_status_label.setStyleSheet(LABEL_STYLE_SUCCESS)

        # start loading the spaCy model
        self._start_spacy_load_task(self.spacy_model)

    def coref_model_changed(self):
        self._start_coref_load_task(self.coref_model)

    def _start_coref_load_task(self, model_name):
        self.cancel_or_await_previous_task()

        # if we don't have a spacy model loaded, do that first
        # note that the task finished handler for the spacy model
        # will start the coreference model loading task again
        if self.nlp is None:
            self._start_spacy_load_task(self.spacy_model)
            return

        # start coreference model loading
        self.information(f"{model_name} loading...")
        self.coref_model_status = f"loading model: {model_name}"
        self.coref_model_status_label.setStyleSheet(LABEL_STYLE_ACTIVE)
        self.setInvalidated(True)
        self.setReady(False)

        # define a worker function to load the coreference model
        def _coref_load_worker(nlp: spacy.language.Language, model_name: str):
            """Load the coreference model."""
            # if we already have a coreference model loaded, unload it first
            if "fastcoref" in nlp.pipe_names:
                nlp.remove_pipe("fastcoref")

            # load new coreference model pipeline component
            if model_name == "fastcoref":
                nlp.add_pipe("fastcoref")
            elif model_name == "lingmess":
                nlp.add_pipe(
                    "fastcoref",
                    config={
                        "model_architecture": "LingMessCoref",
                        "model_path": "biu-nlp/lingmess-coref",
                    },
                )
            else:
                raise ValueError(f"Unknown coreference model: {model_name}")

        self._task = task = Task()
        task.future = self._executor.submit(
            partial(_coref_load_worker, self.nlp, model_name)
        )
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._coref_load_task_finished)

    @pyqtSlot(Future)
    def _coref_load_task_finished(self, future):
        self._assert_task_finished(future)

        # clear task, info message
        self._task = None
        self.Information.clear()
        self.coref_model_status = f"{self.coref_model} ready"
        self.coref_model_status_label.setStyleSheet(LABEL_STYLE_SUCCESS)

        # notify orange that we're ready to receive data
        self.setReady(True)

        # run coref resolution?
        self.update_if_needed()

    def _start_coref_resolution(self):
        if self.corpus is None:
            return

        # if nlp or coref model isn't loaded yet, do that first
        # TODO: we currently start just the nlp/coref load task, then return
        #   should we implement some way of queueing the resolution task?
        self.cancel_or_await_previous_task()

        if self.nlp is None:
            self._start_spacy_load_task(self.spacy_model)
            return

        if not self.nlp.has_pipe("fastcoref"):
            self._start_coref_load_task(self.coref_model)
            return

        def _coref_resolution_worker(corpus: Corpus, nlp: spacy.language.Language, content_attr, callback):
            """Process the corpus with the spaCy pipeline and resolve coreferences."""

            resolved_corpus = corpus.copy()
            coreferences = []
            total = len(corpus)

            # pluck all the texts out of the corpus
            docs = [doc[content_attr].value for doc in corpus]
            config = { "batch_size": 1, "component_cfg": {"fastcoref": {"resolve_text": True}}}

            for doc_idx, doc, resolved_doc in zip(range(len(corpus)), docs, nlp.pipe(docs, **config)):
                resolved_corpus[doc_idx][content_attr] = resolved_doc._.resolved_text

                # Extract coreference clusters
                for cluster_idx, cluster in enumerate(resolved_doc._.coref_clusters):
                    for start, end in cluster:
                        coreferences.append(
                            {
                                "doc": doc_idx,
                                "cluster": cluster_idx,
                                "start": start,
                                "end": end,
                                "text": resolved_doc.text[start:end],
                            }
                        )

                if callback:
                    # update progress
                    callback((doc_idx + 1)/total * 100)

            # re-tokenize the resolved corpus
            resolved_corpus.name = f"{corpus.name} (resolved coreferences)"
            resolved_corpus = BASE_TOKENIZER(resolved_corpus)

            # Build a Table for coreferences, going through a pandas DataFrame for convenience
            _coreferences_df = pd.DataFrame.from_dict(coreferences)
            coreferences = Table.from_pandas_dfs(
                xdf=_coreferences_df[["doc", "cluster", "start", "end"]],
                ydf=_coreferences_df[[]],
                mdf=_coreferences_df[["text"]],
            )

            return resolved_corpus, coreferences

        self._task = task = Task()

        # create an invoke wrapper to safely call QtObject methods across threads
        set_progress = methodinvoke(self, "progressBarSet", (float,))

        # callback handler to check for task cancellation and update progress
        def callback(progress):
            if task.cancelled:
                raise self.TaskCancelledException()
            set_progress(progress)
            
        # Start the coreference resolution task
        self.information("Resolving coreferences...")
        self.progressBarInit() # assumes 100 'total'
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)
        self.stop_button.setEnabled(True)
        self.setReady(False)
        self.setInvalidated(True)
        
        task.future = self._executor.submit(
            partial(_coref_resolution_worker, self.corpus, self.nlp, self.content_attr, callback)
        )
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._coref_resolution_finished)

    def _stop_coref_resolution(self):
        if not self._task: 
            return

        self.stop_button.setEnabled(False)
        self._task.cancelled = True
        self.setReady(True)


    @pyqtSlot(Future)
    def _coref_resolution_finished(self, future: Future):
        """Handle the completion of the coreference resolution task."""
        self._assert_task_finished(future)
        self._task = None  # clear the task
        self.progressBarFinished()
        self.start_button.setVisible(True)
        self.start_button.setDisabled(self.commitOnChange)
        self.stop_button.setVisible(False)
        self.setReady(True)
        self.setInvalidated(False)

        # try getting the result
        try: 
            resolved_corpus, coreferences = future.result()
            self.Outputs.resolved_corpus.send(resolved_corpus)
            self.Outputs.coreferences.send(coreferences)
        except Exception as err:
            if not isinstance(err, self.TaskCancelledException):
                self.error(f"Error during coreference resolution: {err}")
                traceback.print_exc()
        finally: 
            self.Information.clear()

    def handleNewSignals(self): 
        if self.corpus is None: 
            self.Outputs.resolved_corpus.send(None)
            self.Outputs.coreferences.send(None)
            return

        if self.commitOnChange:
            self._start_coref_resolution()
        else:
            self.setInvalidated(True)

    def setInvalidated(self, state: bool):
        # whenever we invalidate, we also want to clear our outputs
        if state:
            self.Outputs.resolved_corpus.send(None)
            self.Outputs.coreferences.send(None)
        super().setInvalidated(state)

    def send_report(self):
        # self.report_plot() includes visualizations in the report
        self.report_caption(self.label)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    # get test data
    corpus = Corpus.from_file("tests/storynavigator-testdata.tab")
    corpus = BASE_TOKENIZER(corpus)  # preprocess the corpus

    WidgetPreview(CoreferenceResolutionWidget).run(
        set_corpus=corpus, no_exit=True
    )  # or any other Table
