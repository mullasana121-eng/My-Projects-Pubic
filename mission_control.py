import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QComboBox, QPushButton,
                             QListWidget, QListWidgetItem, QLabel, QCheckBox)
from PyQt6.QtCore import Qt, QSize

class TaskItemWidget(QWidget):
    """Custom widget to display inside the list for each task."""
    def __init__(self, task_name, priority, list_widget, item):
        super().__init__()
        self.list_widget = list_widget
        self.item = item

        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)

        # Checkbox for completion
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self.toggle_complete)

        # Task Name Label
        self.task_label = QLabel(task_name)
        self.task_label.setStyleSheet("font-size: 15px;")

        # Priority Label with dynamic colors
        self.priority_label = QLabel(f"[{priority}]")
        if priority == "High":
            self.priority_label.setStyleSheet("color: #ff6b6b; font-weight: bold; font-size: 13px;")
        elif priority == "Medium":
            self.priority_label.setStyleSheet("color: #feca57; font-weight: bold; font-size: 13px;")
        else:
            self.priority_label.setStyleSheet("color: #1dd1a1; font-weight: bold; font-size: 13px;")

        # Delete Button
        self.delete_btn = QPushButton("Scrub")
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #ee5253; 
                color: white; 
                border-radius: 4px; 
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b6b;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_task)

        # Add components to the layout
        layout.addWidget(self.checkbox)
        layout.addWidget(self.task_label, stretch=1)
        layout.addWidget(self.priority_label)
        layout.addWidget(self.delete_btn)

        self.setLayout(layout)

    def toggle_complete(self, state):
        """Applies a strikethrough effect when the checkbox is ticked."""
        if state == 2:  # 2 represents the 'Checked' state in PyQt
            self.task_label.setStyleSheet("font-size: 15px; text-decoration: line-through; color: #636e72;")
        else:
            self.task_label.setStyleSheet("font-size: 15px; color: #dfe6e9;")

    def delete_task(self):
        """Removes this item from the parent list widget."""
        row = self.list_widget.row(self.item)
        self.list_widget.takeItem(row)


class TaskTrackerApp(QMainWindow):
    """The main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agency Mission Tracker Pro")
        self.setMinimumSize(550, 650)

        # Apply a rich, dark theme using Qt Style Sheets (QSS)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d3436;
            }
            QLabel {
                color: #dfe6e9;
            }
            QLineEdit {
                background-color: #1e272e;
                color: #dfe6e9;
                border: 1px solid #485460;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #0984e3;
            }
            QComboBox {
                background-color: #1e272e;
                color: #dfe6e9;
                border: 1px solid #485460;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton#addBtn {
                background-color: #0984e3;
                color: #ffffff;
                font-weight: bold;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton#addBtn:hover {
                background-color: #74b9ff;
            }
            QListWidget {
                background-color: #1e272e;
                border: 1px solid #485460;
                border-radius: 5px;
                outline: none;
            }
            QListWidget::item {
                background-color: #2d3436;
                border-bottom: 1px solid #485460;
                padding: 5px;
            }
            QListWidget::item:hover {
                background-color: #353b48;
            }
        """)

        # Set up the central widget and main vertical layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(25, 25, 25, 25)

        # Header Title
        header = QLabel("MISSION OBJECTIVES")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #74b9ff; letter-spacing: 2px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Input Area (Horizontal Layout)
        input_layout = QHBoxLayout()
        
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Enter new objective...")
        
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["Low", "Medium", "High"])
        self.priority_combo.setCurrentText("Medium")
        
        self.add_btn = QPushButton("Add Task")
        self.add_btn.setObjectName("addBtn")
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.clicked.connect(self.add_task)

        # Pressing 'Enter' in the text field also adds the task
        self.task_input.returnPressed.connect(self.add_task)

        input_layout.addWidget(self.task_input, stretch=1)
        input_layout.addWidget(self.priority_combo)
        input_layout.addWidget(self.add_btn)

        layout.addLayout(input_layout)

        # Task List Area
        self.task_list = QListWidget()
        layout.addWidget(self.task_list)

        # Populate with some initial data context
        self.add_task_item("Review Fyers API webhook logic", "High")
        self.add_task_item("Run S3D utility update", "Medium")

    def add_task(self):
        """Triggered when the 'Add Task' button is clicked."""
        task_name = self.task_input.text().strip()
        
        # Don't add empty tasks
        if not task_name:
            return
        
        priority = self.priority_combo.currentText()
        self.add_task_item(task_name, priority)
        
        # Reset input field
        self.task_input.clear()
        self.task_input.setFocus()

    def add_task_item(self, task_name, priority):
        """Creates a custom list item and injects it into the QListWidget."""
        item = QListWidgetItem(self.task_list)
        item.setSizeHint(QSize(0, 60))  # Set height to accommodate our custom widget
        
        # Instantiate the custom widget and assign it to the item
        task_widget = TaskItemWidget(task_name, priority, self.task_list, item)
        self.task_list.setItemWidget(item, task_widget)

# Standard Python execution block
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TaskTrackerApp()
    window.show()
    sys.exit(app.exec())