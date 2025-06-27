from functools import partial
from Orange.data import Table
from Orange.widgets import gui
from concurrent.futures import Future, wait
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from AnyQt.QtCore import Qt, pyqtSlot, QThread
from AnyQt import QtCore

from orangecontrib.text.corpus import Corpus  # type: ignore warning from shared namespace
from orangecontrib.text.preprocess import BASE_TOKENIZER  # type: ignore warning from shared namespace

import spacy
import spacy.cli
import spacy.cli.download

from fastcoref import spacy_component

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
    commitOnChange = Setting(True)  # commit on change of inputs
    coref_model = Setting(
        DEFAULT_FASTCOREF_MODEL
    )  # model to use for coreference resolution
    spacy_model = Setting(DEFAULT_SPACY_MODEL)  # spaCy model to use for tokenization

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
        self.coref_model = DEFAULT_FASTCOREF_MODEL

        self.controlArea.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        self.controlArea.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        # self.controlArea.setLayout(QtWidgets.QHBoxLayout())
        self.groupbox_settings = gui.vBox(self.controlArea, box="Settings")
        self.groupbox_settings.setMinimumSize(QtCore.QSize(250, 0))
        self.groupbox_settings.layout().setAlignment(Qt.AlignmentFlag.AlignTop)

        gui.comboBox(
            self.groupbox_settings,
            self,
            "spacy_model",
            items=SPACY_MODELS,
            label="spaCy Model",
            callback=self.spacy_model_changed,
            sendSelectedValue=True,
        )

        self.spacy_model_status_label = gui.label(self.groupbox_settings, self, "")
        # self.spacy_model_status_label.setStyleSheet()
        # self.spacy_model_status_label.setVisible(False)

        gui.comboBox(
            self.groupbox_settings,
            self,
            "coref_model",
            items=FASTCOREF_MODELS,
            label="Coreference Model",
            callback=self.coref_model_changed,
            sendSelectedValue=True,
        )

        # self.coref_model_status_label = gui.label(self.groupbox_settings, self, "")
        # self.coref_model_status_label.setVisible(False)

        gui.checkBox(
            self.groupbox_settings,
            self,
            "commitOnChange",
            "Commit on change",
            callback=self.setting_changed,
        )

        gui.button(self.groupbox_settings, self, "Run", self._start_coref_resolution)

    @Inputs.corpus
    def set_corpus(self, corpus):
        if corpus:
            self.corpus = corpus
        else:
            self.corpus = None

    def setting_changed(self):
        if self.commitOnChange:
            self.commit()

    def spacy_model_changed(self):
        self._start_spacy_load_task(self.spacy_model)

    def _cancel_or_await_current_task(self):
        """Cancel the current task if it is running, or wait for it to finish."""
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            self._task.watcher.done.disconnect()
            self._task = None

    def _assert_task_finished(self, future):
        """Assert that the current task has finished."""
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is future
        assert future.done()

    def _start_spacy_load_task(self, model_name):
        """Start a task to load the spaCy model."""
        self._cancel_or_await_current_task()

        if not spacy.util.is_package(self.spacy_model):
            self._start_spacy_download_task(self.spacy_model)
            return

        self.Information.spacy_load_model(model_name)
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

        # load the spaCy model
        self.nlp = future.result()

        # start loading the coreference model
        self._start_coref_load_task(self.coref_model)

    def _start_spacy_download_task(self, model_name):
        self._cancel_or_await_current_task()

        self.Information.spacy_download_model(model_name)
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

        # start loading the spaCy model
        self._start_spacy_load_task(self.spacy_model)

    def coref_model_changed(self):
        self._start_coref_load_task(self.coref_model)

    def _start_coref_load_task(self, model_name):
        self._cancel_or_await_current_task()

        # if we don't have a spacy model loaded, do that first
        # note that the task finished handler for the spacy model
        # will start the coreference model loading task again
        if self.nlp is None:
            self._start_spacy_load_task(self.spacy_model)
            return

        # start coreference model loading
        self.information(f"{model_name} loading...")

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

    def _start_coref_resolution(self):
        if self.corpus is None:
            return

        # if nlp or coref model isn't loaded yet, do that first
        # TODO: we currently start just the nlp/coref load task, then return
        #   should we implement some way of queueing the resolution task?
        self._cancel_or_await_current_task()
        if self.nlp is None:
            self._start_spacy_load_task(self.spacy_model)
            return

        if not self.nlp.has_pipe("fastcoref"):
            self._start_coref_load_task(self.coref_model)
            return

        def _coref_resolution_worker(corpus: Corpus, nlp: spacy.language.Language):
            """Process the corpus with the spaCy pipeline and resolve coreferences."""

            resolved_corpus = corpus.copy()
            coreferences = []

            for doc_idx, doc in enumerate(corpus):
                # Process each document in the corpus with the spaCy pipeline
                resolved_doc = nlp(
                    doc["Text"].value,
                    component_cfg={"fastcoref": {"resolve_text": True}},
                )
                resolved_corpus[doc_idx]["Text"] = resolved_doc._.resolved_text

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

        # Start the coreference resolution task
        # TODO: hook up progress bar
        self.information("Resolving coreferences...")
        self._task = task = Task()
        task.future = self._executor.submit(
            partial(_coref_resolution_worker, self.corpus, self.nlp)
        )
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._coref_resolution_finished)

    @pyqtSlot(Future)
    def _coref_resolution_finished(self, future: Future):
        """Handle the completion of the coreference resolution task."""
        self._assert_task_finished(future)
        self._task = None  # clear the task

        resolved_corpus, coreferences = future.result()

        self.Information.clear()
        self.Outputs.resolved_corpus.send(resolved_corpus)
        self.Outputs.coreferences.send(coreferences)

    def commit(self):
        self.Outputs.resolved_corpus.send(self.corpus)
        self.Outputs.coreferences.send(self.corpus)

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
