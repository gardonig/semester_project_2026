from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..core.builder import PosetBuilder
from ..core.config import ASSETS_DIR
from ..core.models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)
from .utils import _relation_verb


class ImagePreviewDialog(QDialog):
    """
    Simple full-window image preview dialog used when the user clicks an image.
    """

    def __init__(self, pixmap: QPixmap, title: str | None = None, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or "Image preview")
        self.setModal(True)
        layout = QVBoxLayout(self)
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Show image at high quality using the original pixmap, scaling down only if needed.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            max_w = int(avail.width() * 0.9)
            max_h = int(avail.height() * 0.9)
        else:
            max_w = pixmap.width()
            max_h = pixmap.height()

        target_w = min(pixmap.width(), max_w)
        target_h = min(pixmap.height(), max_h)
        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        layout.addWidget(label)
        self.resize(scaled.width() + 40, scaled.height() + 40)


class ClickableImageLabel(QLabel):
    """
    QLabel that opens its pixmap in a full-window preview when clicked.
    """

    def __init__(self, preview_title: str | None = None, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self._preview_title = preview_title or "Image preview"
        self._full_pixmap: QPixmap | None = None

    def set_full_pixmap(self, pixmap: QPixmap) -> None:
        """Store the original-resolution pixmap for high-quality preview."""
        self._full_pixmap = pixmap

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            base = self._full_pixmap or self.pixmap()
            if base is None:
                return
            dlg = ImagePreviewDialog(base, self._preview_title, self.window())
            dlg.exec()
        else:
            super().mousePressEvent(event)


class DefinitionDialog(QDialog):
    """
    Shown before the query window. Displays the definition (and examples for vertical).
    User presses "Proceed" to proceed to questions.
    """

    def __init__(self, axis: str = AXIS_VERTICAL) -> None:
        super().__init__()
        self.setWindowTitle("Instructions — Poset Construction")
        self.resize(560, 600)
        self.setModal(True)
        self._axis = axis
        self.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout(self)

        # All text: dark on white (no grey backgrounds)
        _text_style = "color: #1a1a1a; font-size: 14px; padding: 6px 0;"

        # ---- 1. Welcome + anatomical position (side-by-side) ----
        welcome_heading = QLabel("Welcome to the Anatomical Structure Questionnaire")
        welcome_heading.setStyleSheet("color: #1a1a1a; font-weight: bold; font-size: 18px; padding: 0 0 12px 0;")
        layout.addWidget(welcome_heading)

        intro_text = (
            "Thank you for taking part. In this questionnaire you will be asked to compare pairs of "
            "anatomical structures and indicate whether one is strictly above the other (vertical axis) or "
            "strictly to the left of the other (mediolateral axis). Your answers help us build a spatial ordering "
            "that can be used to check and correct automatic segmentations. There are no wrong answers—we need "
            "your clinical judgement.\n\n"
            "We assume the patient is in standard anatomical position, as shown in the figure.\n\n"
            "If you have any questions, please do not hesitate to reach out to Gian or Güney."
        )
        intro_label = QLabel(intro_text)
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(_text_style)

        anatomy_img = ClickableImageLabel("Anatomical axes — full view")
        anatomy_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        anatomy_img.setFixedHeight(400)
        
        # Use ASSETS_DIR from config instead of relative path
        anatomy_path = ASSETS_DIR / "definition_images" / "Anatomy_axes.png"
        
        if anatomy_path.exists():
            anatomy_pix = QPixmap(str(anatomy_path))
            if not anatomy_pix.isNull():
                anatomy_img.set_full_pixmap(anatomy_pix)
                anatomy_img.setPixmap(
                    anatomy_pix.scaledToHeight(400, Qt.SmoothTransformation)
                )
        if anatomy_img.pixmap() is None or anatomy_img.pixmap().isNull():
            anatomy_img.setText("[Anatomical position diagram missing]")
        
        # Put text and image side-by-side
        intro_row = QHBoxLayout()
        intro_row.addWidget(intro_label, stretch=3)
        intro_row.addWidget(anatomy_img, stretch=2)
        layout.addLayout(intro_row)

        layout.addStretch(1)

        # Proceed button + image source note in a bottom bar
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        button_layout = QHBoxLayout(button_box)

        anatomy_ref = QLabel("Images in this window are captured from Complete Anatomy.")
        anatomy_ref.setWordWrap(True)
        anatomy_ref.setStyleSheet("color: #555; font-size: 10px; margin-top: 2px;")
        button_layout.addWidget(anatomy_ref)

        button_layout.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        button_layout.addWidget(proceed_btn)
        layout.addWidget(button_box)


class VerticalDefinitionDialog(QDialog):
    """
    Dedicated window for the vertical 'strictly above' definition and examples.
    Shown only when the vertical axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Definition — Vertical "strictly above"')
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=3)

        heading = QLabel('Vertical relation: "strictly above"')
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            "You will be asked whether one anatomical structure is strictly above another "
            "along the superior–inferior (head–to–toes) axis."
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "Definition: one structure is strictly above another if the lowest point of the upper structure "
            "is higher than the highest point of the lower one."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        q_form_heading = QLabel("What you will be asked")
        q_form_heading.setStyleSheet(_heading_style)
        left_col.addWidget(q_form_heading)

        q_form = QLabel(
            'For each pair of structures you will answer "Yes" or "No" to:\n'
            '  "Is the first structure strictly above the second?"'
        )
        q_form.setWordWrap(True)
        q_form.setStyleSheet(_text_style)
        left_col.addWidget(q_form)

        left_col.addStretch(1)

        # Right column: textual examples + images side by side (No and Yes)
        right_col = QVBoxLayout()
        content.addLayout(right_col, stretch=4)

        _img_dir = ASSETS_DIR / "definition_images"
        img_height = 320

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=2)

        # No example (Femur–Tibia)
        no_col = QVBoxLayout()
        no_col.setSpacing(4)
        examples_col.addLayout(no_col)
        no_text = QLabel('Example 1: "Is the Femur strictly above the Tibia?" → Answer: No.')
        no_text.setWordWrap(True)
        no_text.setStyleSheet(_text_style)
        no_col.addWidget(no_text)

        no_label = ClickableImageLabel("Vertical example 1 — full view")
        no_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_label.setFixedHeight(img_height)
        no_label.setStyleSheet(img_style)
        no_path = _img_dir / "example_vert_No.png"
        if no_path.exists():
            pix = QPixmap(str(no_path))
            if not pix.isNull():
                no_label.set_full_pixmap(pix)
                no_label.setPixmap(
                    pix.scaledToHeight(img_height, Qt.SmoothTransformation)
                )
        if no_label.pixmap() is None or no_label.pixmap().isNull():
            no_label.setText("[Add vertical No example image here]")
            no_label.setStyleSheet(placeholder_style)
        no_col.addWidget(no_label)

        # Yes example (Femur–Fibula)
        yes_col = QVBoxLayout()
        yes_col.setSpacing(4)
        examples_col.addLayout(yes_col)
        yes_text = QLabel('Example 2: "Is the Femur strictly above the Fibula?" → Answer: Yes.')
        yes_text.setWordWrap(True)
        yes_text.setStyleSheet(_text_style)
        yes_text.setContentsMargins(12, 0, 0, 0)
        yes_col.addWidget(yes_text)

        yes_label = ClickableImageLabel("Vertical example 2 — full view")
        yes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        yes_label.setFixedHeight(img_height)
        yes_label.setStyleSheet(img_style)
        yes_path = _img_dir / "eample_vert_Yes.png"
        if yes_path.exists():
            pix = QPixmap(str(yes_path))
            if not pix.isNull():
                yes_label.set_full_pixmap(pix)
                yes_label.setPixmap(
                    pix.scaledToHeight(img_height, Qt.SmoothTransformation)
                )
        if yes_label.pixmap() is None or yes_label.pixmap().isNull():
            yes_label.setText("[Add vertical Yes example image here]")
            yes_label.setStyleSheet(placeholder_style)
        yes_col.addWidget(yes_label)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=1)

        com_heading = QLabel("Center of mass (CoM)")
        com_heading.setStyleSheet(_heading_style)
        com_col.addWidget(com_heading)

        com_text = QLabel(
            "For the vertical axis, CoM is scaled from 0 (toes/feet, most inferior) "
            "to 100 (vertex/head, most superior)."
        )
        com_text.setWordWrap(True)
        com_text.setStyleSheet(_text_style)
        com_col.addWidget(com_text)

        com_img = ClickableImageLabel("Vertical CoM — full view")
        com_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        com_img.setFixedHeight(2 * img_height)
        com_img.setStyleSheet(img_style)
        # Labeled image for vertical CoM
        com_path = _img_dir / "CoM_vert_axis.png"
        if com_path.exists():
            pix = QPixmap(str(com_path))
            if not pix.isNull():
                com_img.set_full_pixmap(pix)
                com_img.setPixmap(
                    pix.scaledToHeight(2 * img_height, Qt.SmoothTransformation)
                )
        if com_img.pixmap() is None or com_img.pixmap().isNull():
            com_img.setText("[Add vertical CoM image here]")
            com_img.setStyleSheet(placeholder_style)
        com_col.addWidget(com_img)

        # Proceed button + image source note below all content (bottom bar)
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        btn_row = QHBoxLayout(button_box)
        source = QLabel("Images in this window are captured from Complete Anatomy.")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        btn_row.addWidget(source)
        btn_row.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row.addWidget(proceed_btn)
        main.addWidget(button_box)


class MediolateralDefinitionDialog(QDialog):
    """
    Dedicated window for the lateral (right–left) 'strictly to the left of' definition and examples.
    Shown only when the lateral axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Definition — Lateral "strictly to the left of"')
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=3)

        heading = QLabel('Lateral relation: "strictly to the left of"')
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            "You will be asked whether one anatomical structure is strictly to the left of another "
            "along the right–left (lateral) axis, from the patient's perspective."
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "Definition: one structure is strictly to the left of another if the rightmost point of the first "
            "is to the left of the leftmost point of the second."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        patient_note = QLabel(
            "Left and right are always defined from the patient's view: "
            "the patient's right femur is to the right of the patient's left femur."
        )
        patient_note.setWordWrap(True)
        patient_note.setStyleSheet(_text_style)
        left_col.addWidget(patient_note)

        q_form_heading = QLabel("What you will be asked")
        q_form_heading.setStyleSheet(_heading_style)
        left_col.addWidget(q_form_heading)

        q_form = QLabel(
            'For each pair of structures you will answer "Yes" or "No" to:\n'
            '  "Is the first structure strictly to the left of the second?"'
        )
        q_form.setWordWrap(True)
        q_form.setStyleSheet(_text_style)
        left_col.addWidget(q_form)

        left_col.addStretch(1)

        # Right column: two examples side-by-side (text above image)
        right_col = QVBoxLayout()
        content.addLayout(right_col, stretch=4)

        _img_dir = ASSETS_DIR / "definition_images"
        img_height = 280

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=2)

        # Example (Yes)
        yes_col = QVBoxLayout()
        yes_col.setSpacing(4)
        examples_col.addLayout(yes_col)
        ex1_text = QLabel(
            'Example 1: "Is the Left femur strictly to the left of the Right femur?" '
            "→ Answer: Yes (from the patient's perspective)."
        )
        ex1_text.setWordWrap(True)
        ex1_text.setStyleSheet(_text_style)
        yes_col.addWidget(ex1_text)

        ex1_img = ClickableImageLabel("Lateral example 1 — full view")
        ex1_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex1_img.setFixedHeight(img_height)
        ex1_img.setStyleSheet(img_style)
        ex1_path = _img_dir / "example_lat_yes.png"
        if ex1_path.exists():
            pix = QPixmap(str(ex1_path))
            if not pix.isNull():
                ex1_img.set_full_pixmap(pix)
                ex1_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex1_img.pixmap() is None or ex1_img.pixmap().isNull():
            ex1_img.setText("[Add mediolateral Yes example image here]")
            ex1_img.setStyleSheet(placeholder_style)
        yes_col.addWidget(ex1_img)

        # Example (No)
        no_col = QVBoxLayout()
        no_col.setSpacing(4)
        examples_col.addLayout(no_col)
        ex2_text = QLabel(
            'Example 2: "Is the Left femur strictly to the left of the pelvis?" '
            "→ Answer: No (they overlap in the mediolateral direction)."
        )
        ex2_text.setWordWrap(True)
        ex2_text.setStyleSheet(_text_style)
        ex2_text.setContentsMargins(12, 0, 0, 0)
        no_col.addWidget(ex2_text)

        ex2_img = ClickableImageLabel("Lateral example 2 — full view")
        ex2_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex2_img.setFixedHeight(img_height)
        ex2_img.setStyleSheet(img_style)
        ex2_path = _img_dir / "example_lat_no.png"
        if ex2_path.exists():
            pix = QPixmap(str(ex2_path))
            if not pix.isNull():
                ex2_img.set_full_pixmap(pix)
                ex2_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex2_img.pixmap() is None or ex2_img.pixmap().isNull():
            ex2_img.setText("[Add mediolateral No example image here]")
            ex2_img.setStyleSheet(placeholder_style)
        no_col.addWidget(ex2_img)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=1)

        com_heading = QLabel("Center of mass (CoM)")
        com_heading.setStyleSheet(_heading_style)
        com_col.addWidget(com_heading)

        com_text = QLabel(
            "For the lateral axis, CoM is scaled from 0 (far right, e.g. right thumb) "
            "to 100 (far left, e.g. left thumb), from the patient's perspective."
        )
        com_text.setWordWrap(True)
        com_text.setStyleSheet(_text_style)
        com_col.addWidget(com_text)

        com_img = ClickableImageLabel("Lateral CoM — full view")
        com_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        com_img.setFixedHeight(2 * img_height)
        com_img.setStyleSheet(img_style)
        # Labeled image for lateral CoM
        com_path = _img_dir / "CoM_lat_axis.png"
        if com_path.exists():
            pix = QPixmap(str(com_path))
            if not pix.isNull():
                com_img.set_full_pixmap(pix)
                com_img.setPixmap(pix.scaledToHeight(2 * img_height, Qt.SmoothTransformation))
        if com_img.pixmap() is None or com_img.pixmap().isNull():
            com_img.setText("[Add mediolateral CoM image here]")
            com_img.setStyleSheet(placeholder_style)
        com_col.addWidget(com_img)

        # Proceed button + image source note below all content
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        btn_row = QHBoxLayout(button_box)
        source = QLabel("Images in this window are captured from Complete Anatomy.")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        btn_row.addWidget(source)
        btn_row.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row.addWidget(proceed_btn)
        main.addWidget(button_box)


class AnteroposteriorDefinitionDialog(QDialog):
    """
    Dedicated window for the anteroposterior (front–back) 'strictly in front of' definition and examples.
    Shown only when the anteroposterior axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Definition — Anteroposterior "strictly in front of"')
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=3)

        heading = QLabel('Anteroposterior relation: "strictly in front of"')
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            "You will be asked whether one anatomical structure is strictly in front of another "
            "along the front–back (anteroposterior) axis."
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "Definition: one structure is strictly in front of another if the posterior-most point of the first "
            "is anterior to the anterior-most point of the second."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        q_form_heading = QLabel("What you will be asked")
        q_form_heading.setStyleSheet(_heading_style)
        left_col.addWidget(q_form_heading)

        q_form = QLabel(
            'For each pair of structures you will answer "Yes" or "No" to:\n'
            '  "Is the first structure strictly in front of the second?"'
        )
        q_form.setWordWrap(True)
        q_form.setStyleSheet(_text_style)
        left_col.addWidget(q_form)

        left_col.addStretch(1)

        # Right column: two examples side-by-side (text above image)
        right_col = QVBoxLayout()
        content.addLayout(right_col, stretch=4)

        img_height = 280

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        _img_dir = ASSETS_DIR / "definition_images"

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=2)

        # Example (Yes)
        yes_col = QVBoxLayout()
        yes_col.setSpacing(4)
        examples_col.addLayout(yes_col)
        ex1_text = QLabel(
            'Example 1 (Yes): "Is the sternum strictly in front of the thoracic spine?" → Answer: Yes.'
        )
        ex1_text.setWordWrap(True)
        ex1_text.setStyleSheet(_text_style)
        yes_col.addWidget(ex1_text)

        ex1_img = ClickableImageLabel("Anteroposterior example 1 — full view")
        ex1_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex1_img.setFixedHeight(img_height)
        ex1_img.setStyleSheet(img_style)
        ex1_path = _img_dir / "example_ap_yes.png"
        if ex1_path.exists():
            pix = QPixmap(str(ex1_path))
            if not pix.isNull():
                ex1_img.set_full_pixmap(pix)
                ex1_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex1_img.pixmap() is None or ex1_img.pixmap().isNull():
            ex1_img.setText("[Add anteroposterior Yes example image here]")
            ex1_img.setStyleSheet(placeholder_style)
        yes_col.addWidget(ex1_img)

        # Example (No)
        no_col = QVBoxLayout()
        no_col.setSpacing(4)
        examples_col.addLayout(no_col)
        ex2_text = QLabel(
            'Example 2 (No): "Is the clavicle strictly in front of the cervical spine?" → Answer: No.'
        )
        ex2_text.setWordWrap(True)
        ex2_text.setStyleSheet(_text_style)
        ex2_text.setContentsMargins(12, 0, 0, 0)
        no_col.addWidget(ex2_text)

        ex2_img = ClickableImageLabel("Anteroposterior example 2 — full view")
        ex2_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex2_img.setFixedHeight(img_height)
        ex2_img.setStyleSheet(img_style)
        ex2_path = _img_dir / "example_ap_no.png"
        if ex2_path.exists():
            pix = QPixmap(str(ex2_path))
            if not pix.isNull():
                ex2_img.set_full_pixmap(pix)
                ex2_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex2_img.pixmap() is None or ex2_img.pixmap().isNull():
            ex2_img.setText("[Add anteroposterior No example image here]")
            ex2_img.setStyleSheet(placeholder_style)
        no_col.addWidget(ex2_img)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=1)

        com_heading = QLabel("Center of mass (CoM)")
        com_heading.setStyleSheet(_heading_style)
        com_col.addWidget(com_heading)

        com_text = QLabel(
            "For the anteroposterior axis, CoM is scaled from 0 (back/dorsal side) "
            "to 100 (front/ventral side)."
        )
        com_text.setWordWrap(True)
        com_text.setStyleSheet(_text_style)
        com_col.addWidget(com_text)

        com_img = ClickableImageLabel("Anteroposterior CoM — full view")
        com_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        com_img.setFixedHeight(2 * img_height)
        com_img.setStyleSheet(img_style)
        # Labeled image for anteroposterior CoM
        com_path = _img_dir / "CoM_AP_axis.png"
        if com_path.exists():
            pix = QPixmap(str(com_path))
            if not pix.isNull():
                com_img.set_full_pixmap(pix)
                com_img.setPixmap(pix.scaledToHeight(2 * img_height, Qt.SmoothTransformation))
        if com_img.pixmap() is None or com_img.pixmap().isNull():
            com_img.setText("[Add anteroposterior CoM image here]")
            com_img.setStyleSheet(placeholder_style)
        com_col.addWidget(com_img)

        # Proceed button + image source note below all content
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        btn_row = QHBoxLayout(button_box)
        source = QLabel("Images in this window are captured from Complete Anatomy.")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        btn_row.addWidget(source)
        btn_row.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row.addWidget(proceed_btn)
        main.addWidget(button_box)


class QueryDialog(QDialog):
    """
    Standalone dialog for expert queries only.
    Clinicians focus on answering questions; no structure input.
    """

    def __init__(
        self,
        poset_builder: PosetBuilder,
        autosave_path: Path,
        axis: str,
        save_callback: Callable[[str, List[Structure], Set[Tuple[int, int]]], None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("Expert Query")
        self.resize(520, 420)
        self.setModal(False)

        self.poset_builder = poset_builder
        self._autosave_path = autosave_path
        self._axis = axis
        self._save_callback = save_callback
        self.pending_pair: Tuple[int, int] | None = None
        # answer is True (Yes), False (No), or None ("Not sure"/skipped)
        self._answer_history: List[Tuple[int, int, Optional[bool]]] = []

        # For vertical axis, detect bilateral (Left/Right) cores to combine in question text
        self._bilateral_cores: Set[str] = set()
        if self._axis == AXIS_VERTICAL:
            core_counts: Dict[str, int] = {}
            names = [s.name.strip() for s in self.poset_builder.structures]
            for name in names:
                if name.startswith("Left "):
                    core = name[5:].strip()
                elif name.startswith("Right "):
                    core = name[6:].strip()
                else:
                    continue
                core_counts[core] = core_counts.get(core, 0) + 1
            self._bilateral_cores = {c for c, cnt in core_counts.items() if cnt >= 2}

        layout = QVBoxLayout(self)

        # Question card
        self.question_card = QFrame()
        self.question_card.setMinimumHeight(160)
        self.question_card.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 24px;
                margin: 12px 0;
            }
            """
        )
        card_layout = QVBoxLayout(self.question_card)
        self.query_label = QLabel("")
        self.query_label.setWordWrap(True)
        self.query_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.query_label.setStyleSheet(
            "color: #1a1a1a; font-size: 22px; font-weight: 500; line-height: 1.4;"
        )
        card_layout.addWidget(self.query_label)
        layout.addWidget(self.question_card)

        # Back, Yes, No
        btn_row = QHBoxLayout()
        self.back_btn = QPushButton("← Undo")
        self.back_btn.setEnabled(False)
        self.back_btn.setStyleSheet("padding: 10px 16px; font-size: 14px;")
        self.back_btn.clicked.connect(self.go_back_one_question)
        btn_row.addWidget(self.back_btn)

        self.yes_btn = QPushButton("Yes")
        self.yes_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2e7d32; color: white; border: none; border-radius: 8px;
                padding: 14px 28px; font-size: 18px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #388e3c; }
            QPushButton:pressed:enabled { background-color: #1b5e20; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.yes_btn.clicked.connect(lambda: self.answer_query(True))

        self.no_btn = QPushButton("No")
        self.no_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #c62828; color: white; border: none; border-radius: 8px;
                padding: 14px 28px; font-size: 18px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #d32f2f; }
            QPushButton:pressed:enabled { background-color: #b71c1c; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.no_btn.clicked.connect(lambda: self.answer_query(False))
        
        self.not_sure_btn = QPushButton("Not sure")
        self.not_sure_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f2f2f7; color: #1a1a1a; border: 1px solid #d1d1d6; border-radius: 8px;
                padding: 14px 18px; font-size: 16px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #e5e5ea; }
            QPushButton:pressed:enabled { background-color: #d1d1d6; }
            QPushButton:disabled { background-color: #f2f2f7; color: #757575; border: 1px solid #e0e0e0; }
            """
        )
        self.not_sure_btn.clicked.connect(lambda: self.answer_query(None))
        btn_row.addStretch()
        btn_row.addWidget(self.no_btn)
        btn_row.addWidget(self.not_sure_btn)
        btn_row.addWidget(self.yes_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #e0e0e0;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: #007aff;
            }
            """
        )
        layout.addWidget(self.progress_bar)

        # Finish and Close (shown when done)
        self.finish_btn = QPushButton("Done")
        self.finish_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        self.finish_btn.clicked.connect(self.accept)
        self.finish_btn.hide()
        layout.addWidget(self.finish_btn)

        self._advance_to_next_query()

    def _autosave_poset(self) -> None:
        if not self._autosave_path or not self._save_callback:
            return
        try:
            structures, edges = self.poset_builder.get_final_relations()
            self._save_callback(self._axis, structures, edges)
        except Exception:
            pass

    def _advance_to_next_query(self) -> None:
        pair = self.poset_builder.next_pair()
        self.pending_pair = pair
        if pair is None:
            self._autosave_poset()
            self.query_label.setText(
                "Thank you for your participation!\n\nEnjoy the pizza 🍕"
            )
            self.yes_btn.hide()
            self.no_btn.hide()
            self.not_sure_btn.hide()
            self.back_btn.hide()
            self.finish_btn.show()
            self.progress_bar.setValue(100)
            return
        i, j = pair
        si, sj = self.poset_builder.structures[i], self.poset_builder.structures[j]
        verb = _relation_verb(self._axis)
        name_i = self._display_name(i, si.name)
        name_j = self._display_name(j, sj.name)
        self.query_label.setText(f"Is/Are the {name_i} {verb} the {name_j}?")
        self._update_progress()

    def answer_query(self, is_above: Optional[bool]) -> None:
        if self.pending_pair is None:
            return
        i, j = self.pending_pair
        self._answer_history.append((i, j, is_above))
        self.back_btn.setEnabled(True)
        if is_above is True:
            self.poset_builder.record_response(i, j, True)
        elif is_above is None:
            self.poset_builder.record_skip(i, j)
        self._autosave_poset()
        self._advance_to_next_query()

    def go_back_one_question(self) -> None:
        if not self._answer_history:
            return
        last_i, last_j, last_answer = self._answer_history.pop()
        if last_answer is True:
            self.poset_builder.edges.discard((last_i, last_j))
        elif last_answer is None:
            self.poset_builder.unskip_pair(last_i, last_j)
        self.poset_builder.finished = False
        self.poset_builder.current_gap = last_j - last_i
        self.poset_builder.current_i = last_i + 1
        self.pending_pair = (last_i, last_j)
        si, sj = self.poset_builder.structures[last_i], self.poset_builder.structures[last_j]
        verb = _relation_verb(self._axis)
        name_i = self._display_name(last_i, si.name)
        name_j = self._display_name(last_j, sj.name)
        self.query_label.setText(f"(Correcting) Is/Are the {name_i} {verb} the {name_j}?")
        self.yes_btn.setEnabled(True)
        self.no_btn.setEnabled(True)
        self.not_sure_btn.setEnabled(True)
        if not self._answer_history:
            self.back_btn.setEnabled(False)
        self._update_progress()
        self._autosave_poset()

    def _update_progress(self) -> None:
        asked = len(self._answer_history)
        remaining = self.poset_builder.estimate_remaining_questions()
        total = asked + remaining
        if total == 0:
            value = 0
        else:
            value = int(100 * asked / total)
        self.progress_bar.setValue(value)

    def _display_name(self, idx: int, original: str) -> str:
        if self._axis != AXIS_VERTICAL:
            return original

        name = original.strip()
        core = None
        if name.startswith("Left "):
            core = name[5:].strip()
        elif name.startswith("Right "):
            core = name[6:].strip()

        if core and core in self._bilateral_cores:
            return core
        return original