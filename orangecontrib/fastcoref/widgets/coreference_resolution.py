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
        self._spacy_download_task = None
        self._spacy_load_task = None
        self._coref_load_task = None
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
        if spacy.util.is_package(self.spacy_model):
            self._start_spacy_load_task(self.spacy_model)
        else:
            self._start_spacy_download_task(self.spacy_model)

    def _start_spacy_load_task(self, model_name):
        if self._spacy_load_task is not None:
            self._spacy_load_task.cancel()
            assert self._spacy_load_task.future.done()
            self._spacy_load_task.watcher.done.disconnect(
                self._spacy_load_task_finished
            )
            self._spacy_load_task = None

        self.Information.spacy_load_model(model_name)
        self._spacy_load_task = task = Task()
        task.future = self._executor.submit(spacy.load, model_name)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._spacy_load_task_finished)

    @pyqtSlot(Future)
    def _spacy_load_task_finished(self, future):
        # assert that we're looking at the correct task, and that it has finished
        assert self.thread() is QThread.currentThread()
        assert self._spacy_load_task is not None
        assert self._spacy_load_task.future is future
        assert future.done()

        # clear task, info message
        self._spacy_load_task = None
        self.Information.spacy_load_model.clear()

        # load the spaCy model
        self.nlp = future.result()

        # start loading the coreference model
        self._start_coref_load_task(self.coref_model)

    def _start_spacy_download_task(self, model_name):
        if self._spacy_download_task is not None:
            self._spacy_download_task.cancel()
            assert self._spacy_download_task.future.done()
            self._spacy_download_task.watcher.done.disconnect(
                self._spacy_download_task_finished
            )
            self._spacy_download_task = None

        self.Information.spacy_download_model(model_name)
        self._spacy_download_task = task = Task()
        task.future = self._executor.submit(spacy.cli.download, model_name)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._spacy_download_task_finished)

    @pyqtSlot(Future)
    def _spacy_download_task_finished(self, future):
        # assert that we're looking at the correct task, and that it has finished
        assert self.thread() is QThread.currentThread()
        assert self._spacy_download_task is not None
        assert self._spacy_download_task.future is future
        assert future.done()

        # clear task, info message
        self._spacy_download_task = None
        self.Information.spacy_download_model.clear()

        # start loading the spaCy model
        self._start_spacy_load_task(self.spacy_model)

    def coref_model_changed(self):
        if self.nlp is not None:
            self._start_coref_load_task(self.coref_model)

    def _start_coref_load_task(self, model_name):
        if self._coref_load_task is not None:
            self._coref_load_task.cancel()
            assert self._coref_load_task.future.done()
            self._coref_load_task.watcher.done.disconnect(
                self._coref_load_task_finished
            )
            self._coref_load_task = None

        self.information(f"{model_name} loading...")

        self._coref_load_task = task = Task()

        if model_name == "fastcoref":
            task.future = self._executor.submit(self.nlp.add_pipe, "fastcoref")
        elif model_name == "lingmess":
            task.future = self._executor.submit(
                self.nlp.add_pipe,
                "fastcoref",
                config={
                    "model_architecture": "LingMessCoref",
                    "model_path": "biu-nlp/lingmess-coref",
                },
            )
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._coref_load_task_finished)

    @pyqtSlot(Future)
    def _coref_load_task_finished(self, future):
        # assert that we're looking at the correct task, and that it has finished
        assert self.thread() is QThread.currentThread()
        assert self._coref_load_task is not None
        assert self._coref_load_task.future is future
        assert future.done()

        # clear task, info message
        self._coref_load_task = None
        self.Information.clear()

    def _start_coref_resolution(self):
        if self.corpus is None or self.nlp is None:
            return

        # Process the corpus with the spaCy pipeline
        docs = [
            self.nlp(
                doc["Text"].value, component_cfg={"fastcoref": {"resolve_text": True}}
            )
            for doc in self.corpus
        ]
        newdocs = []

        # Extract mentions and coreferences
        coreferences = []
        for doc_idx, doc in enumerate(docs):
            newdocs.append(doc._.resolved_text)
            for cluster_idx, cluster in enumerate(doc._.coref_clusters):
                for start, end in cluster:
                    coref = {
                        "doc": doc_idx,
                        "cluster": cluster_idx,
                        "start": start,
                        "end": end,
                        "text": doc.text[start:end],
                    }
                    coreferences.append(coref)

        # Create a copy of the original corpus with resolved texts, and rerun tokenization
        _resolved_corpus = self.corpus.copy()
        for i, doc in enumerate(newdocs):
            _resolved_corpus[i]["Text"] = doc
        _resolved_corpus.name = f"{self.corpus.name} (resolved)"
        _resolved_corpus = BASE_TOKENIZER(_resolved_corpus)
        self.Outputs.resolved_corpus.send(_resolved_corpus)

        # Build a Table for coreferences, going through a pandas DataFrame for convenience
        coreferences_df = pd.DataFrame.from_dict(coreferences)
        coreferences_table = Table.from_pandas_dfs(
            xdf=coreferences_df[["doc", "cluster", "start", "end"]],
            ydf=coreferences_df[[]],
            mdf=coreferences_df[["text"]],
        )
        self.Outputs.coreferences.send(coreferences_table)

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
