import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableView, QCheckBox,
                            QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
                            QLabel, QAbstractItemView, QComboBox, QSpinBox,
                            QGroupBox, QScrollArea, QTabWidget, QTextEdit,
                            QMessageBox, QLineEdit, QDockWidget)
from PyQt5.QtCore import QAbstractTableModel, Qt, QTimer, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# Add api_lib to path
sys.path.append(os.path.join(os.getcwd(),"lib"))
from api_lib import Robot, get_robots, render_full_map

class Worker(QObject):
    """
    Worker thread for long-running operations.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    robot_data_updated = pyqtSignal(dict)
    map_updated = pyqtSignal(str)
    pois_updated = pyqtSignal(object)
    path_planned = pyqtSignal(dict)

    def __init__(self, robot):
        super().__init__()
        self.robot = robot

    def refresh_all(self):
        try:
            self.robot.refresh()
            state = self.robot.get_state()
            self.robot_data_updated.emit(state)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Failed to refresh robot data: {e}")

    def update_map(self, temp_map_path):
        try:
            map_file = render_full_map(self.robot.SN, out_png=str(temp_map_path))
            self.map_updated.emit(str(map_file))
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Failed to update map: {e}")

    def update_pois(self):
        try:
            pois_df = self.robot.get_pois()
            self.pois_updated.emit(pois_df)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Failed to load POIs: {e}")
    
    def plan_path(self, poi_name, temp_map_path):
        try:
            result = self.robot.plan_path(poi_name, out_png=str(temp_map_path))
            self.path_planned.emit(result)
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"Failed to plan path: {e}")

class RobotMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Monitor")
        self.setGeometry(100, 100, 1600, 900)
        self.robot = None
        self.temp_map_path = Path("temp_robot_map.png")
        self.worker = None
        self.thread = None
        self.init_ui()
        self.load_robots()

    def init_ui(self):
        self.create_robot_selection_dock()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Robot info and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(500)

        # Robot state table
        state_group = QGroupBox("Robot State")
        state_layout = QVBoxLayout(state_group)
        self.state_table = QTableView()
        self.state_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        state_layout.addWidget(self.state_table)
        left_layout.addWidget(state_group)

        # Position info (relPos/isAt)
        pos_group = QGroupBox("Position & Proximity")
        pos_layout = QVBoxLayout(pos_group)
        self.pos_text = QTextEdit()
        self.pos_text.setReadOnly(True)
        self.pos_text.setMaximumHeight(150)
        pos_layout.addWidget(self.pos_text)
        left_layout.addWidget(pos_group)

        # Control buttons
        self.control_group = QGroupBox("Robot Control")
        control_layout = QVBoxLayout(self.control_group)

        # Navigation controls
        nav_layout = QHBoxLayout()
        self.charge_btn = QPushButton("Go Charge")
        self.charge_btn.clicked.connect(self.go_charge)
        nav_layout.addWidget(self.charge_btn)

        self.back_btn = QPushButton("Go Back")
        self.back_btn.clicked.connect(self.go_back)
        nav_layout.addWidget(self.back_btn)
        control_layout.addLayout(nav_layout)

        # POI navigation
        poi_layout = QHBoxLayout()
        poi_layout.addWidget(QLabel("Go to POI:"))
        self.poi_combo = QComboBox()
        self.poi_combo.addItem("Select POI...")
        poi_layout.addWidget(self.poi_combo)
        self.go_poi_btn = QPushButton("Go")
        self.go_poi_btn.clicked.connect(self.go_to_selected_poi)
        poi_layout.addWidget(self.go_poi_btn)
        control_layout.addLayout(poi_layout)

        # Wait action
        wait_layout = QHBoxLayout()
        wait_layout.addWidget(QLabel("Wait at POI:"))
        self.wait_poi_combo = QComboBox()
        self.wait_poi_combo.addItem("Select POI...")
        wait_layout.addWidget(self.wait_poi_combo)
        self.wait_seconds = QSpinBox()
        self.wait_seconds.setRange(1, 300)
        self.wait_seconds.setValue(10)
        self.wait_seconds.setSuffix(" sec")
        wait_layout.addWidget(self.wait_seconds)
        self.wait_btn = QPushButton("Wait")
        self.wait_btn.clicked.connect(self.wait_at_poi)
        wait_layout.addWidget(self.wait_btn)
        control_layout.addLayout(wait_layout)

        # Pickup/Dropdown controls
        pickup_layout = QHBoxLayout()
        pickup_layout.addWidget(QLabel("Shelf:"))
        self.shelf_combo = QComboBox()
        self.shelf_combo.addItem("Select Shelf...")
        pickup_layout.addWidget(self.shelf_combo)
        self.pickup_btn = QPushButton("Pickup")
        self.pickup_btn.clicked.connect(self.pickup_shelf)
        pickup_layout.addWidget(self.pickup_btn)
        self.dropdown_btn = QPushButton("Dropdown")
        self.dropdown_btn.clicked.connect(self.dropdown_shelf)
        pickup_layout.addWidget(self.dropdown_btn)
        control_layout.addLayout(pickup_layout)

        # Path planning
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Plan Path to:"))
        self.path_poi_combo = QComboBox()
        self.path_poi_combo.addItem("Select POI...")
        path_layout.addWidget(self.path_poi_combo)
        self.plan_path_btn = QPushButton("Plan & Show")
        self.plan_path_btn.clicked.connect(self.plan_and_show_path)
        path_layout.addWidget(self.plan_path_btn)
        control_layout.addLayout(path_layout)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh Now")
        self.refresh_btn.clicked.connect(self.refresh_all)
        control_layout.addWidget(self.refresh_btn)

        left_layout.addWidget(self.control_group)

        # Task info
        task_group = QGroupBox("Current Task")
        task_layout = QVBoxLayout(task_group)
        self.task_text = QTextEdit()
        self.task_text.setReadOnly(True)
        self.task_text.setMaximumHeight(100)
        task_layout.addWidget(self.task_text)
        self.cancel_task_btn = QPushButton("Cancel Task")
        self.cancel_task_btn.clicked.connect(self.cancel_current_task)
        task_layout.addWidget(self.cancel_task_btn)
        left_layout.addWidget(task_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Right panel - Map view with tabs
        right_panel = QTabWidget()

        # Map tab
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)

        # Map controls
        map_ctrl_layout = QHBoxLayout()
        self.show_overlays_cb = QCheckBox("Show Overlays")
        self.show_overlays_cb.setChecked(True)
        self.show_overlays_cb.stateChanged.connect(self.update_map)
        map_ctrl_layout.addWidget(self.show_overlays_cb)

        self.show_robot_cb = QCheckBox("Show Robot")
        self.show_robot_cb.setChecked(True)
        self.show_robot_cb.stateChanged.connect(self.update_map)
        map_ctrl_layout.addWidget(self.show_robot_cb)

        self.update_map_btn = QPushButton("Update Map")
        self.update_map_btn.clicked.connect(self.update_map)
        map_ctrl_layout.addWidget(self.update_map_btn)
        map_ctrl_layout.addStretch()
        map_layout.addLayout(map_ctrl_layout)

        # Map display (scrollable)
        scroll = QScrollArea()
        self.map_label = QLabel()
        self.map_label.setStyleSheet("background-color: #1a1a1a;")
        self.map_label.setScaledContents(False)
        scroll.setWidget(self.map_label)
        scroll.setWidgetResizable(True)
        map_layout.addWidget(scroll)

        right_panel.addTab(map_widget, "Live Map")

        # POI table tab
        poi_widget = QWidget()
        poi_layout = QVBoxLayout(poi_widget)
        self.poi_table = QTableView()
        self.poi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        poi_layout.addWidget(self.poi_table)
        right_panel.addTab(poi_widget, "POIs")

        # Areas table tab
        areas_widget = QWidget()
        areas_layout = QVBoxLayout(areas_widget)
        self.areas_table = QTableView()
        self.areas_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        areas_layout.addWidget(self.areas_table)
        right_panel.addTab(areas_widget, "Areas")

        main_layout.addWidget(right_panel, stretch=1)
        
        self.control_group.setEnabled(False)

    def create_robot_selection_dock(self):
        dock = QDockWidget("Robot Selection", self)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        
        dock_widget = QWidget()
        dock_layout = QVBoxLayout(dock_widget)

        # Filter controls
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Business Name:"))
        self.business_filter = QLineEdit()
        self.business_filter.setPlaceholderText("Filter by business name...")
        self.business_filter.textChanged.connect(self.filter_changed)
        filter_layout.addWidget(self.business_filter)

        self.online_filter = QCheckBox("Show Only Online Robots")
        self.online_filter.stateChanged.connect(self.filter_changed)
        filter_layout.addWidget(self.online_filter)

        dock_layout.addWidget(filter_group)

        # Robot table
        self.robot_table = QTableView()
        self.robot_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.robot_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.robot_table.setSortingEnabled(True)
        self.robot_table.doubleClicked.connect(self.select_robot)
        dock_layout.addWidget(self.robot_table)

        # Action buttons
        button_layout = QHBoxLayout()
        self.refresh_robots_button = QPushButton("Refresh List")
        self.refresh_robots_button.clicked.connect(self.load_robots)
        button_layout.addWidget(self.refresh_robots_button)
        
        self.select_robot_button = QPushButton("Select Robot")
        self.select_robot_button.clicked.connect(self.select_robot)
        button_layout.addWidget(self.select_robot_button)
        dock_layout.addLayout(button_layout)
        
        dock.setWidget(dock_widget)

    def load_robots(self):
        try:
            self.robots_df = get_robots()
            for col in ['robotId', 'model', 'isOnLine', 'business_name']:
                if col not in self.robots_df.columns:
                    self.robots_df[col] = "N/A"
            
            self.base_model = PandasModel(self.robots_df[['robotId', 'model', 'isOnLine', 'business_name']])
            self.robot_table.setModel(self.base_model)
            self.robot_table.resizeColumnsToContents()
            self.robot_table.setSortingEnabled(True)
            self.filter_changed()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load robots: {str(e)}")
            self.robots_df = None

    def filter_changed(self):
        if self.robots_df is None:
            return

        business_filter_text = self.business_filter.text().lower()
        online_only = self.online_filter.isChecked()

        filtered_df = self.robots_df.copy()

        if business_filter_text:
            filtered_df = filtered_df[filtered_df['business_name'].str.lower().str.contains(business_filter_text, na=False)]

        if online_only:
            if filtered_df['isOnLine'].dtype == 'bool':
                filtered_df = filtered_df[filtered_df['isOnLine'] == True]
            else:
                filtered_df = filtered_df[filtered_df['isOnLine'].str.lower() == 'true']

        filtered_model = PandasModel(filtered_df[['robotId', 'model', 'isOnLine', 'business_name']])
        self.robot_table.setModel(filtered_model)
        self.robot_table.resizeColumnsToContents()

    def select_robot(self):
        selected_rows = self.robot_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "Please select a robot from the table.")
            return

        model = self.robot_table.model()
        selected_row_index = selected_rows[0].row()
        robot_sn = model._df.iloc[selected_row_index]['robotId']

        try:
            self.robot = Robot(robot_sn)
            self.setWindowTitle(f"Robot Monitor - {self.robot.SN}")
            self.temp_map_path = Path(f"temp_robot_map_{self.robot.SN}.png")
            self.control_group.setEnabled(True)
            self.start_worker()
            self.refresh_all()
            self.update_pois()
            self.update_map()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize robot monitor for {robot_sn}: {str(e)}")

    def start_worker(self):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        self.thread = QThread()
        self.worker = Worker(self.robot)
        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.show_error)
        self.worker.robot_data_updated.connect(self.update_robot_data_ui)
        self.worker.map_updated.connect(self.update_map_ui)
        self.worker.pois_updated.connect(self.update_poi_combos_ui)
        self.worker.path_planned.connect(self.update_path_ui)

        self.thread.start()

    def show_error(self, error_message):
        QMessageBox.warning(self, "Error", error_message)

    def refresh_all(self):
        if self.robot and self.worker:
            self.worker.refresh_all()

    def update_map(self):
        if self.robot and self.worker:
            self.worker.update_map(self.temp_map_path)
            
    def update_pois(self):
        if self.robot and self.worker:
            self.worker.update_pois()

    def update_robot_data_ui(self, state):
        try:
            pose = state.get("pose", (None, None, None))
            task = state.get("task", None)

            data = [
                ["Serial Number", self.robot.SN],
                ["Business", self.robot.business],
                ["Online", str(state.get("isOnLine", False))],
                ["Charging", str(state.get("isCharging", False))],
                ["Battery", f"{state.get('battery', 'N/A')}%"],
                ["Position (x, y)", f"({pose[0]:.2f}, {pose[1]:.2f})" if pose[0] is not None else "N/A"],
                ["Orientation (yaw)", f"{pose[2]:.2f}°" if pose[2] is not None else "N/A"],
                ["Speed", f"{state.get('speed', 'N/A')} m/s"],
                ["Emergency Stop", str(state.get("isEmergencyStop", False))],
                ["Obstruction", str(state.get("hasObstruction", False))],
                ["Errors", str(state.get("errors", []))]
            ]

            self.state_model = TableModel(data)
            self.state_table.setModel(self.state_model)
            self.state_table.resizeColumnsToContents()

            is_at = state.get("isAt")
            if is_at is not None and not is_at.empty:
                pos_info = "Currently At:\n"
                for _, row in is_at.iterrows():
                    pos_info += f"  • {row.get('name', 'Unknown')} ({row.get('kind', 'N/A')} - {row.get('type', 'N/A')})\n"
            else:
                pos_info = "Not at any specific location"
            self.pos_text.setPlainText(pos_info)

            if task:
                task_info = f"Task ID: {task.get('taskId', 'N/A')}\n"
                task_info += f"Name: {task.get('name', 'N/A')}\n"
                task_info += f"Status: {task.get('state', 'N/A')}"
                self.task_text.setPlainText(task_info)
                self.cancel_task_btn.setEnabled(True)
            else:
                self.task_text.setPlainText("No active task")
                self.cancel_task_btn.setEnabled(False)

        except Exception as e:
            self.show_error(f"Failed to update robot data UI: {e}")

    def update_poi_combos_ui(self, pois_df):
        try:
            for combo in [self.poi_combo, self.wait_poi_combo, self.path_poi_combo]:
                while combo.count() > 1:
                    combo.removeItem(1)

            self.shelf_combo.clear()
            self.shelf_combo.addItem("Select Shelf...")

            if not pois_df.empty:
                poi_names = pois_df['name'].tolist()
                for name in poi_names:
                    self.poi_combo.addItem(name)
                    self.wait_poi_combo.addItem(name)
                    self.path_poi_combo.addItem(name)

                shelves = pois_df[pois_df['type'] == 34]['name'].tolist()
                for shelf in shelves:
                    self.shelf_combo.addItem(shelf)

                self.poi_model = PandasModel(pois_df)
                self.poi_table.setModel(self.poi_model)
                self.poi_table.resizeColumnsToContents()
        except Exception as e:
            self.show_error(f"Failed to load POIs: {e}")

    def update_map_ui(self, map_file):
        try:
            pixmap = QPixmap(map_file)
            if not pixmap.isNull():
                scaled = pixmap.scaled(1200, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.map_label.setPixmap(scaled)
                self.map_label.adjustSize()
            else:
                self.map_label.setText("Failed to load map image")
        except Exception as e:
            self.show_error(f"Failed to update map UI: {e}")
            self.map_label.setText(f"Map Error: {e}")
            
    def update_path_ui(self, result):
        try:
            length = result.get("length_m", 0)
            pixmap = QPixmap(str(self.temp_map_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(1200, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.map_label.setPixmap(scaled)
                self.map_label.adjustSize()

            QMessageBox.information(self, "Path Planned",
                                  f"Path planned\nLength: {length:.2f}m")
        except Exception as e:
            self.show_error(f"Failed to display path: {e}")

    def go_charge(self):
        if self.robot:
            try:
                self.robot.go_charge()
                QMessageBox.information(self, "Success", "Charging task created")
            except Exception as e:
                self.show_error(f"Failed to create charging task: {e}")

    def go_back(self):
        if self.robot:
            try:
                self.robot.go_back()
                QMessageBox.information(self, "Success", "Return task created")
            except Exception as e:
                self.show_error(f"Failed to create return task: {e}")

    def go_to_selected_poi(self):
        if self.robot:
            poi_name = self.poi_combo.currentText()
            if poi_name == "Select POI...":
                QMessageBox.warning(self, "Warning", "Please select a POI")
                return
            try:
                self.robot.go_to_poi(poi_name)
                QMessageBox.information(self, "Success", f"Navigation to {poi_name} started")
            except Exception as e:
                self.show_error(f"Failed to navigate: {e}")

    def wait_at_poi(self):
        if self.robot:
            poi_name = self.wait_poi_combo.currentText()
            if poi_name == "Select POI...":
                QMessageBox.warning(self, "Warning", "Please select a POI")
                return
            try:
                seconds = self.wait_seconds.value()
                self.robot.wait_at(poi_name, seconds)
                QMessageBox.information(self, "Success", f"Wait task created at {poi_name} for {seconds}s")
            except Exception as e:
                self.show_error(f"Failed to create wait task: {e}")

    def pickup_shelf(self):
        if self.robot:
            shelf_name = self.shelf_combo.currentText()
            if shelf_name == "Select Shelf...":
                QMessageBox.warning(self, "Warning", "Please select a shelf")
                return
            try:
                self.robot.pickup_at(shelf_name)
                QMessageBox.information(self, "Success", f"Pickup task created at {shelf_name}")
            except Exception as e:
                self.show_error(f"Failed to create pickup task: {e}")

    def dropdown_shelf(self):
        if self.robot:
            shelf_name = self.shelf_combo.currentText()
            if shelf_name == "Select Shelf...":
                QMessageBox.warning(self, "Warning", "Please select a shelf")
                return
            try:
                self.robot.dropdown_at(shelf_name)
                QMessageBox.information(self, "Success", f"Dropdown task created at {shelf_name}")
            except Exception as e:
                self.show_error(f"Failed to create dropdown task: {e}")

    def plan_and_show_path(self):
        if self.robot and self.worker:
            poi_name = self.path_poi_combo.currentText()
            if poi_name == "Select POI...":
                QMessageBox.warning(self, "Warning", "Please select a POI")
                return
            self.worker.plan_path(poi_name, self.temp_map_path)

    def cancel_current_task(self):
        if self.robot:
            try:
                self.robot.cancel_task()
                QMessageBox.information(self, "Success", "Task cancelled")
                self.refresh_all()
            except Exception as e:
                self.show_error(f"Failed to cancel task: {e}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        if self.temp_map_path.exists():
            try:
                self.temp_map_path.unlink()
            except:
                pass
        event.accept()


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return 2

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._data[index.row()][index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return ["Attribute", "Value"][section]
        return None


class PandasModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            value = self._df.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(self._df.index[section])
        return None


def main():
    app = QApplication(sys.argv)
    main_window = RobotMonitorApp()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
