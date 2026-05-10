from __future__ import annotations

from PyQt6 import QtWidgets


def show_scrollable_text(parent: QtWidgets.QWidget, title: str, message: str, details: str) -> None:
    dialog = ScrollableTextDialog(parent, title, message, details, confirm=False)
    dialog.exec()


def confirm_scrollable_text(parent: QtWidgets.QWidget, title: str, message: str, details: str) -> bool:
    dialog = ScrollableTextDialog(parent, title, message, details, confirm=True)
    return dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted


class ScrollableTextDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None, title: str, message: str, details: str, confirm: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 460)
        root = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(message)
        label.setWordWrap(True)
        root.addWidget(label)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.details.setPlainText(details)
        root.addWidget(self.details, 1)

        buttons = QtWidgets.QDialogButtonBox()
        if confirm:
            buttons.setStandardButtons(
                QtWidgets.QDialogButtonBox.StandardButton.Yes | QtWidgets.QDialogButtonBox.StandardButton.No
            )
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Yes).setText("Tak")
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.No).setText("Nie")
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
        else:
            buttons.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.reject)
        root.addWidget(buttons)


def scrollable_details_line_wrap_mode() -> str:
    return "WidgetWidth"
