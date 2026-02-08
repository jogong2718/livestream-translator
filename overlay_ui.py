"""
overlay_ui.py
A transparent, always-on-top PyQt6 overlay window that displays the latest
translated text at the bottom of the screen.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class OverlayWindow(QWidget):
    """
    Frameless, transparent, always-on-top subtitle overlay.
    Sits at the bottom-centre of the primary screen.
    """

    # Custom signal so any thread can safely update the UI
    update_text_signal = pyqtSignal(str, str)  # (translated_text, romaji_text)

    # ------------------------------------------------------------------ init
    def __init__(self, parent=None):
        super().__init__(parent)

        # ----- window flags -----
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # hides from taskbar / dock
            | Qt.WindowType.WindowTransparentForInput  # click-through
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)

        # ----- layout -----
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 20)

        # Translation label (main line – English output)
        self.label = self._make_label(size=22, bold=True)
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Romaji label (smaller, shown below main text when available)
        self.romaji_label = self._make_label(size=15, bold=False, color="#cccccc")
        self.romaji_label.hide()
        layout.addWidget(self.romaji_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

        # Connect the cross-thread signal
        self.update_text_signal.connect(self._on_update_text)

        # Position at screen bottom
        self._reposition()

        # Always visible: no fade timer

    # -------------------------------------------------------- public helpers

    def set_text(self, translated: str, romaji: str = ""):
        """
        Thread-safe method to update overlay text.
        Can be called from any thread – emits a signal handled on the UI thread.
        """
        self.update_text_signal.emit(translated, romaji)

    # -------------------------------------------------------- internal slots

    @pyqtSlot(str, str)
    def _on_update_text(self, translated: str, romaji: str):
        self.label.setText(translated)
        self.label.adjustSize()

        if romaji:
            self.romaji_label.setText(romaji)
            self.romaji_label.adjustSize()
            self.romaji_label.show()
        else:
            self.romaji_label.hide()

        # Ensure overlay stays visible
        self.setWindowOpacity(1.0)
        self.show()

        # Resize / reposition to fit new text
        self._reposition()

    # -------------------------------------------------------- positioning

    def _reposition(self):
        """Place overlay at the bottom-centre of the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geom = screen.availableGeometry()

        # Resize to fit text
        self.adjustSize()

        # Use a fixed wide overlay so multiline romaji never clips
        max_w = int(geom.width() * 0.6)
        w = max_w
        margins = self.layout().contentsMargins()
        inner_w = max(0, w - (margins.left() + margins.right()))
        self.label.setFixedWidth(inner_w)
        self.romaji_label.setFixedWidth(inner_w)
        h = self.sizeHint().height()

        x = geom.x() + (geom.width() - w) // 2
        y = geom.y() + geom.height() - h - 60  # 60 px above bottom edge
        self.setGeometry(x, y, w, h)

    # -------------------------------------------------------- painting

    def paintEvent(self, event):
        """Draw a semi-transparent dark rounded-rect background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 180))  # black @ ~70% opacity
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 16, 16)
        painter.end()

    # -------------------------------------------------------- widget factory

    @staticmethod
    def _make_label(size: int = 20, bold: bool = True, color: str = "#ffffff") -> QLabel:
        label = QLabel("")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        font = QFont("Helvetica Neue", size)
        font.setBold(bold)
        label.setFont(font)

        label.setStyleSheet(f"color: {color}; background: transparent;")

        # Soft drop shadow for readability
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(1, 1)
        label.setGraphicsEffect(shadow)

        return label
