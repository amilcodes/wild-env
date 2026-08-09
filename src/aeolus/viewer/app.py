"""Qt/VTK desktop application for replay inspection."""

from __future__ import annotations

import os
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from aeolus.replay.render import render_frame_2d, render_frame_3d
from aeolus.viewer.config import LayerConfig, ViewerConfig
from aeolus.viewer.imagery import load_imagery
from aeolus.viewer.model import ReplayModel
from aeolus.viewer.scene2d import BACKGROUND, draw_operational_map

if TYPE_CHECKING:
    from aeolus.replay.recorder import ReplayBundle


def _qt() -> dict[str, Any]:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install aeolus-ia[viewer] to use the native desktop viewer") from exc
    return {
        "Qt": QtCore.Qt,
        "QTimer": QtCore.QTimer,
        "QAction": QtGui.QAction,
        "QKeySequence": QtGui.QKeySequence,
        **{
            name: getattr(QtWidgets, name)
            for name in (
                "QApplication",
                "QCheckBox",
                "QComboBox",
                "QDockWidget",
                "QFileDialog",
                "QHeaderView",
                "QHBoxLayout",
                "QLabel",
                "QMainWindow",
                "QMessageBox",
                "QPushButton",
                "QSlider",
                "QStackedWidget",
                "QTableWidget",
                "QTableWidgetItem",
                "QToolBar",
                "QVBoxLayout",
                "QWidget",
            )
        },
    }


def run_viewer(
    replay: ReplayBundle,
    config: ViewerConfig,
    *,
    start_frame: int = 0,
) -> int:
    """Start the native desktop viewer and block until its window closes."""

    qt = _qt()
    QApplication = qt["QApplication"]
    application = QApplication.instance() or QApplication([])
    window = _build_window(qt, replay, config, start_frame=start_frame)
    window.show()
    return int(application.exec())


def _build_window(
    qt: dict[str, Any],
    replay: ReplayBundle,
    config: ViewerConfig,
    *,
    start_frame: int,
):
    Qt = qt["Qt"]
    QTimer = qt["QTimer"]
    QAction = qt["QAction"]
    QKeySequence = qt["QKeySequence"]
    QCheckBox = qt["QCheckBox"]
    QComboBox = qt["QComboBox"]
    QDockWidget = qt["QDockWidget"]
    QFileDialog = qt["QFileDialog"]
    QHeaderView = qt["QHeaderView"]
    QHBoxLayout = qt["QHBoxLayout"]
    QLabel = qt["QLabel"]
    QMainWindow = qt["QMainWindow"]
    QMessageBox = qt["QMessageBox"]
    QPushButton = qt["QPushButton"]
    QSlider = qt["QSlider"]
    QStackedWidget = qt["QStackedWidget"]
    QTableWidget = qt["QTableWidget"]
    QTableWidgetItem = qt["QTableWidgetItem"]
    QToolBar = qt["QToolBar"]
    QVBoxLayout = qt["QVBoxLayout"]
    QWidget = qt["QWidget"]

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    model = ReplayModel(replay)
    imagery = load_imagery(model, config.imagery)

    class MapCanvas(FigureCanvasQTAgg):
        def __init__(self):
            self.figure = Figure(facecolor=BACKGROUND, layout="constrained")
            self.axis = self.figure.add_subplot(111)
            super().__init__(self.figure)
            self.mpl_connect("motion_notify_event", self._mouse_move)
            self.on_cursor = None

        def _mouse_move(self, event) -> None:
            if (
                self.on_cursor is not None
                and event.inaxes is self.axis
                and event.xdata is not None
                and event.ydata is not None
            ):
                self.on_cursor(float(event.xdata), float(event.ydata))

        def update_scene(
            self,
            frame: int,
            scene_config: ViewerConfig,
            selected: str | None,
        ) -> None:
            draw_operational_map(
                self.axis,
                model,
                scene_config,
                frame,
                selected_resource=selected,
                imagery=imagery,
            )
            minute = int(model.minutes[frame])
            self.axis.set_title(
                f"{model.title}  ·  T+{minute:03d} min",
                loc="left",
                color="#e8edf0",
                fontsize=11,
                weight="medium",
                pad=8,
            )
            if scene_config.camera.mode == "follow" and selected is not None:
                resource = model.resource(frame, selected)
                radius = scene_config.camera.follow_radius_cells
                self.axis.set_xlim(resource["x"] - radius, resource["x"] + radius)
                self.axis.set_ylim(resource["y"] + radius, resource["y"] - radius)
            self.draw_idle()

    class TerrainCanvas(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            self.available = False
            self.plotter = None
            self.error = None
            try:
                if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
                    raise RuntimeError("VTK is disabled under Qt's offscreen test platform")
                import pyvista as pv
                from pyvistaqt import QtInteractor

                self.pv = pv
                self.plotter = QtInteractor(self)
                self.plotter.set_background(BACKGROUND)
                self.plotter.enable_anti_aliasing("ssaa")
                layout.addWidget(self.plotter.interactor)
                self.available = True
            except Exception as exc:  # pragma: no cover - platform OpenGL path
                self.error = exc
                message = QLabel(
                    "The 3D view requires the viewer extra and an OpenGL 3.3 context.\n"
                    "Install with: python -m pip install -e '.[viewer]'"
                )
                message.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(message)
            self._actor_names: list[str] = []

        def update_scene(
            self,
            frame: int,
            scene_config: ViewerConfig,
            selected: str | None,
        ) -> None:
            if not self.available or self.plotter is None:
                return
            plotter = self.plotter
            plotter.clear()
            elevation = model.static("static/elevation_m")
            height, width = elevation.shape
            y, x = np.mgrid[0:height, 0:width]
            cell_size = model.cell_size_m
            x_m = x.astype(float) * cell_size
            y_m = y.astype(float) * cell_size
            z = (elevation - float(elevation.min())) * scene_config.camera.vertical_exaggeration
            grid = self.pv.StructuredGrid(x_m, y_m, z)
            phase = model.field("truth/phase", frame)
            intensity = model.field("truth/intensity_kw_m", frame)
            normalized = (elevation - elevation.min()) / max(
                float(np.ptp(elevation)),
                1.0,
            )
            terrain_rgb = np.stack(
                (
                    0.10 + 0.48 * normalized,
                    0.18 + 0.42 * normalized,
                    0.12 + 0.30 * normalized,
                ),
                axis=-1,
            )
            if scene_config.layers.imagery and imagery is not None:
                rgb = (
                    scene_config.imagery.opacity * imagery.rgb
                    + (1.0 - scene_config.imagery.opacity) * terrain_rgb
                )
            else:
                rgb = terrain_rgb
            burned = phase == 2
            rgb[burned] *= 0.22
            active = phase == 1
            fire = np.clip(np.log1p(intensity) / np.log(20_001.0), 0.0, 1.0)
            rgb[active, 0] = 1.0
            rgb[active, 1] = 0.15 + 0.70 * fire[active]
            rgb[active, 2] = 0.02
            grid.point_data["rgb"] = np.clip(
                rgb.reshape(-1, 3, order="F") * 255.0,
                0,
                255,
            ).astype(np.uint8)
            plotter.add_mesh(
                grid,
                scalars="rgb",
                rgb=True,
                smooth_shading=True,
                ambient=0.32,
                diffuse=0.68,
                specular=0.05,
            )
            if scene_config.layers.active_fire and active.any():
                active_y, active_x = np.where(active)
                points = np.column_stack(
                    (
                        active_x * cell_size,
                        active_y * cell_size,
                        z[active_y, active_x] + max(float(np.ptp(z)) * 0.02, 2.0),
                    )
                )
                cloud = self.pv.PolyData(points)
                cloud["intensity"] = intensity[active_y, active_x]
                plotter.add_mesh(
                    cloud,
                    scalars="intensity",
                    cmap="inferno",
                    log_scale=True,
                    point_size=7,
                    render_points_as_spheres=True,
                    show_scalar_bar=False,
                )
            for layer_enabled, field_name, color, offset in (
                (
                    scene_config.layers.water,
                    "treatment/water_coverage_gpc",
                    "#168fe5",
                    0.045,
                ),
                (
                    scene_config.layers.retardant,
                    "treatment/retardant_coverage_gpc",
                    "#d91b78",
                    0.055,
                ),
            ):
                if not layer_enabled or not model.has(field_name):
                    continue
                treatment = model.field(field_name, frame)
                treatment_y, treatment_x = np.where(treatment > 0.05)
                if treatment_x.size:
                    points = np.column_stack(
                        (
                            treatment_x * cell_size,
                            treatment_y * cell_size,
                            z[treatment_y, treatment_x] + cell_size * offset,
                        )
                    )
                    plotter.add_mesh(
                        self.pv.PolyData(points),
                        color=color,
                        point_size=5.0,
                        render_points_as_spheres=True,
                        opacity=0.82,
                    )
            if scene_config.layers.constructed_line and model.has("treatment/line_status"):
                line_status = model.field("treatment/line_status", frame)
                for status, color in (
                    (1, "#64cfff"),
                    (2, "#58e078"),
                    (3, "#ff4f52"),
                ):
                    line_y, line_x = np.where(line_status == status)
                    if line_x.size:
                        points = np.column_stack(
                            (
                                line_x * cell_size,
                                line_y * cell_size,
                                z[line_y, line_x] + cell_size * 0.065,
                            )
                        )
                        plotter.add_mesh(
                            self.pv.PolyData(points),
                            color=color,
                            point_size=6.0,
                            render_points_as_spheres=True,
                        )
            resources = model.resources(frame)
            for resource_index, resource in enumerate(resources):
                start_minute = int(model.minutes[frame]) - scene_config.playback.trail_minutes
                start = int(np.searchsorted(model.minutes, start_minute))
                track_x = np.asarray(model.states["resources/x"][start : frame + 1, resource_index])
                track_y = np.asarray(model.states["resources/y"][start : frame + 1, resource_index])
                ix = np.clip(np.rint(track_x).astype(int), 0, width - 1)
                iy = np.clip(np.rint(track_y).astype(int), 0, height - 1)
                display_offset = (
                    max(cell_size * 0.04, 1.0)
                    if resource["kind"] == "crew"
                    else max(float(np.ptp(z)) * 0.055, cell_size * 0.28)
                )
                points = np.column_stack(
                    (
                        track_x * cell_size,
                        track_y * cell_size,
                        z[iy, ix] + display_offset,
                    )
                )
                color = {
                    "retardant": "#f4d58d",
                    "water": "#57b9ff",
                    "sensor": "#d9c75c",
                    "crew": "#76d98c",
                }[resource["kind"]]
                if len(points) > 1:
                    plotter.add_lines(
                        points,
                        color=color,
                        width=2.0,
                        connected=True,
                    )
                plotter.add_mesh(
                    self.pv.Sphere(
                        radius=(cell_size * 0.32 if resource["id"] == selected else cell_size * 0.22),
                        center=points[-1],
                    ),
                    color=color,
                    smooth_shading=True,
                )
            for site in model.service_sites:
                site_x, site_y = int(site["x"]), int(site["y"])
                plotter.add_mesh(
                    self.pv.Cylinder(
                        center=(
                            site_x * cell_size,
                            site_y * cell_size,
                            z[site_y, site_x] + cell_size * 0.08,
                        ),
                        direction=(0.0, 0.0, 1.0),
                        radius=cell_size * 0.30,
                        height=cell_size * 0.16,
                    ),
                    color="#d7e1e6",
                )
            plotter.add_text(
                f"{model.title}  ·  T+{int(model.minutes[frame]):03d} min",
                position="upper_left",
                font_size=10,
                color="#e8edf0",
            )
            plotter.add_text(
                "Aircraft height is a display offset",
                position="lower_left",
                font_size=7,
                color="#8f9ba3",
            )
            center = np.array(
                [
                    (width - 1) * cell_size / 2.0,
                    (height - 1) * cell_size / 2.0,
                    (float(z.min()) + float(z.max())) / 2.0,
                ]
            )
            if scene_config.camera.mode == "follow" and selected is not None:
                resource = model.resource(frame, selected)
                ix = int(np.clip(round(resource["x"]), 0, width - 1))
                iy = int(np.clip(round(resource["y"]), 0, height - 1))
                center = np.array(
                    [
                        resource["x"] * cell_size,
                        resource["y"] * cell_size,
                        z[iy, ix],
                    ]
                )
            horizontal_span = float(np.hypot((width - 1) * cell_size, (height - 1) * cell_size))
            radius = max(horizontal_span * 1.25, float(np.ptp(z)) * 2.5)
            azimuth = np.deg2rad(scene_config.camera.azimuth_deg)
            elevation_angle = np.deg2rad(scene_config.camera.elevation_deg)
            camera = center + radius * np.array(
                [
                    np.cos(elevation_angle) * np.cos(azimuth),
                    np.cos(elevation_angle) * np.sin(azimuth),
                    np.sin(elevation_angle),
                ]
            )
            plotter.camera_position = [camera.tolist(), center.tolist(), [0.0, 0.0, 1.0]]
            plotter.camera.view_angle = 28.0
            plotter.reset_camera_clipping_range()
            plotter.render()

    class ViewerWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.model = model
            self.config = config
            self.frame = int(np.clip(start_frame, 0, model.frame_count - 1))
            self.selected_resource = model.resource_ids[0] if model.resource_ids else None
            self.playing = False
            self.frame_accumulator = 0.0
            self.setWindowTitle(f"Aeolus replay — {model.title}")
            self.resize(config.window.width, config.window.height)
            self.stack = QStackedWidget()
            self.map_canvas = MapCanvas()
            self.map_canvas.on_cursor = self._cursor_moved
            self.terrain_canvas = TerrainCanvas()
            self.stack.addWidget(self.map_canvas)
            self.stack.addWidget(self.terrain_canvas)
            self.setCentralWidget(self.stack)
            self._build_toolbar()
            self._build_timeline()
            self._build_layers()
            self._build_vehicles()
            self._build_events()
            self._build_menu()
            self.timer = QTimer(self)
            self.timer.setInterval(round(1000 / self.config.playback.refresh_hz))
            self.timer.timeout.connect(self._tick)
            self.timer.start()
            self._set_view(self.config.window.start_view)
            self._update_scene()

        def _build_toolbar(self) -> None:
            toolbar = QToolBar("Replay", self)
            toolbar.setMovable(False)
            self.addToolBar(toolbar)
            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self._toggle_play)
            toolbar.addWidget(self.play_button)
            first = QPushButton("|<")
            first.clicked.connect(lambda: self._set_frame(0))
            toolbar.addWidget(first)
            previous = QPushButton("<")
            previous.clicked.connect(lambda: self._set_frame(self.frame - 1))
            toolbar.addWidget(previous)
            following = QPushButton(">")
            following.clicked.connect(lambda: self._set_frame(self.frame + 1))
            toolbar.addWidget(following)
            last = QPushButton(">|")
            last.clicked.connect(lambda: self._set_frame(model.frame_count - 1))
            toolbar.addWidget(last)
            toolbar.addSeparator()
            toolbar.addWidget(QLabel("View "))
            self.view_selector = QComboBox()
            self.view_selector.addItem("Operational 2D", "operational_2d")
            self.view_selector.addItem("Terrain 3D", "terrain_3d")
            self.view_selector.currentIndexChanged.connect(
                lambda _: self._set_view(self.view_selector.currentData())
            )
            toolbar.addWidget(self.view_selector)
            toolbar.addWidget(QLabel("  Camera "))
            self.camera_selector = QComboBox()
            self.camera_selector.addItem("Incident", "incident")
            self.camera_selector.addItem("North up", "north_up")
            self.camera_selector.addItem("Follow selected", "follow")
            self.camera_selector.currentIndexChanged.connect(self._camera_changed)
            toolbar.addWidget(self.camera_selector)
            toolbar.addWidget(QLabel("  Rate "))
            self.rate_selector = QComboBox()
            for rate in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64):
                self.rate_selector.addItem(f"{rate:g} min/s", float(rate))
            rate_index = self.rate_selector.findData(float(self.config.playback.rate))
            self.rate_selector.setCurrentIndex(max(rate_index, 0))
            toolbar.addWidget(self.rate_selector)

        def _build_timeline(self) -> None:
            dock = QDockWidget("Time", self)
            dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
            container = QWidget()
            layout = QHBoxLayout(container)
            self.time_label = QLabel()
            self.time_label.setMinimumWidth(310)
            self.timeline = QSlider(Qt.Orientation.Horizontal)
            self.timeline.setRange(0, model.frame_count - 1)
            self.timeline.setValue(self.frame)
            self.timeline.valueChanged.connect(self._set_frame)
            layout.addWidget(self.time_label)
            layout.addWidget(self.timeline, 1)
            dock.setWidget(container)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        def _build_layers(self) -> None:
            self.layer_dock = QDockWidget("Layers", self)
            container = QWidget()
            layout = QVBoxLayout(container)
            self.layer_checks: dict[str, Any] = {}
            for item in fields(LayerConfig):
                checkbox = QCheckBox(item.name.replace("_", " ").capitalize())
                checkbox.setChecked(bool(getattr(self.config.layers, item.name)))
                checkbox.toggled.connect(lambda checked, name=item.name: self._layer_changed(name, checked))
                self.layer_checks[item.name] = checkbox
                layout.addWidget(checkbox)
            layout.addStretch(1)
            self.layer_dock.setWidget(container)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.layer_dock)
            self.layer_dock.setVisible(self.config.window.show_layer_panel)

        def _build_vehicles(self) -> None:
            self.vehicle_dock = QDockWidget("Vehicles", self)
            container = QWidget()
            layout = QVBoxLayout(container)
            self.vehicle_table = QTableWidget(len(model.resource_ids), 6)
            self.vehicle_table.setHorizontalHeaderLabels(
                ["Vehicle", "Status", "Task", "Payload", "Endurance", "Site"]
            )
            self.vehicle_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )
            self.vehicle_table.horizontalHeader().setStretchLastSection(True)
            self.vehicle_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            self.vehicle_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.vehicle_table.itemSelectionChanged.connect(self._vehicle_selected)
            self.vehicle_detail = QLabel()
            self.vehicle_detail.setWordWrap(True)
            layout.addWidget(self.vehicle_table, 1)
            layout.addWidget(self.vehicle_detail)
            self.vehicle_dock.setWidget(container)
            self.vehicle_dock.setMinimumWidth(520)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.vehicle_dock)
            self.vehicle_dock.setVisible(self.config.window.show_vehicle_panel)

        def _build_events(self) -> None:
            self.event_dock = QDockWidget("Events", self)
            self.event_table = QTableWidget(len(model.events), 3)
            self.event_table.setHorizontalHeaderLabels(["Minute", "Event", "Resource / site"])
            self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self.event_table.horizontalHeader().setStretchLastSection(True)
            self.event_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            self.event_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            for row, event in enumerate(model.events):
                detail = (
                    event.payload.get("resource")
                    or event.payload.get("site")
                    or event.payload.get("source")
                    or ""
                )
                for column, value in enumerate((event.minute, event.kind, detail)):
                    self.event_table.setItem(row, column, QTableWidgetItem(str(value)))
            self.event_table.cellDoubleClicked.connect(
                lambda row, _column: self._set_frame(model.frame_for_minute(model.events[row].minute))
            )
            self.event_dock.setWidget(self.event_table)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.event_dock)
            self.event_dock.setVisible(self.config.window.show_event_panel)

        def _build_menu(self) -> None:
            file_menu = self.menuBar().addMenu("&File")
            export_action = QAction("Export current frame…", self)
            export_action.setShortcut(QKeySequence("Ctrl+E"))
            export_action.triggered.connect(self._export_frame)
            file_menu.addAction(export_action)
            save_config = QAction("Save view configuration…", self)
            save_config.triggered.connect(self._save_config)
            file_menu.addAction(save_config)
            close_action = QAction("Close", self)
            close_action.setShortcut(QKeySequence("Ctrl+W"))
            close_action.triggered.connect(self.close)
            file_menu.addAction(close_action)
            view_menu = self.menuBar().addMenu("&View")
            for label, shortcut, name in (
                ("Operational 2D", "1", "operational_2d"),
                ("Terrain 3D", "2", "terrain_3d"),
            ):
                action = QAction(label, self)
                action.setShortcut(QKeySequence(shortcut))
                action.triggered.connect(lambda _checked=False, view=name: self._set_view(view))
                view_menu.addAction(action)
            play_action = QAction("Play / pause", self)
            play_action.setShortcut(QKeySequence("Space"))
            play_action.triggered.connect(self._toggle_play)
            self.addAction(play_action)
            previous_action = QAction("Previous frame", self)
            previous_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
            previous_action.triggered.connect(lambda: self._set_frame(self.frame - 1))
            self.addAction(previous_action)
            next_action = QAction("Next frame", self)
            next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
            next_action.triggered.connect(lambda: self._set_frame(self.frame + 1))
            self.addAction(next_action)

        def _set_view(self, name: str) -> None:
            index = 1 if name == "terrain_3d" else 0
            self.stack.setCurrentIndex(index)
            selector_index = self.view_selector.findData(name)
            if self.view_selector.currentIndex() != selector_index:
                self.view_selector.blockSignals(True)
                self.view_selector.setCurrentIndex(selector_index)
                self.view_selector.blockSignals(False)
            self._update_scene()

        def _toggle_play(self) -> None:
            self.playing = not self.playing
            self.play_button.setText("Pause" if self.playing else "Play")

        def _tick(self) -> None:
            if not self.playing:
                return
            rate = float(self.rate_selector.currentData())
            self.frame_accumulator += rate / self.config.playback.refresh_hz
            advance = int(self.frame_accumulator)
            if advance < 1:
                return
            self.frame_accumulator -= advance
            target = self.frame + advance
            if target >= model.frame_count:
                if self.config.playback.loop:
                    target %= model.frame_count
                else:
                    target = model.frame_count - 1
                    self.playing = False
                    self.play_button.setText("Play")
            self._set_frame(target)

        def _set_frame(self, frame: int) -> None:
            value = int(np.clip(frame, 0, model.frame_count - 1))
            if value == self.frame and self.timeline.value() == value:
                return
            self.frame = value
            self.timeline.blockSignals(True)
            self.timeline.setValue(value)
            self.timeline.blockSignals(False)
            self._update_scene()

        def _camera_changed(self) -> None:
            mode = str(self.camera_selector.currentData())
            camera = replace(
                self.config.camera,
                mode=mode,
                follow_resource=self.selected_resource if mode == "follow" else None,
                azimuth_deg=-90.0 if mode == "north_up" else self.config.camera.azimuth_deg,
                elevation_deg=90.0 if mode == "north_up" else self.config.camera.elevation_deg,
            )
            self.config = replace(self.config, camera=camera)
            self._update_scene()

        def _layer_changed(self, name: str, checked: bool) -> None:
            self.config = replace(
                self.config,
                layers=replace(self.config.layers, **{name: checked}),
            )
            self._update_scene()

        def _vehicle_selected(self) -> None:
            rows = self.vehicle_table.selectionModel().selectedRows()
            if not rows:
                return
            self.selected_resource = model.resource_ids[rows[0].row()]
            if self.config.camera.mode == "follow":
                self.config = replace(
                    self.config,
                    camera=replace(
                        self.config.camera,
                        follow_resource=self.selected_resource,
                    ),
                )
            self._update_scene()

        def _cursor_moved(self, x: float, y: float) -> None:
            if not (0 <= x < model.shape[1] and 0 <= y < model.shape[0]):
                return
            condition = model.conditions(self.frame, x, y)
            world = model.grid_to_world(x, y)
            coordinate = (
                f"x {x * model.cell_size_m / 1000:.2f} km, y {y * model.cell_size_m / 1000:.2f} km"
                if world is None
                else f"{world[0]:.1f}, {world[1]:.1f} {model.spatial_reference.get('crs')}"
            )
            self.statusBar().showMessage(
                f"{coordinate}  ·  wind {condition['wind_direction_deg']:.0f}°/"
                f"{condition['wind_speed_m_s']:.1f} m s⁻¹  ·  "
                f"{condition['air_temperature_c']:.1f} °C  ·  "
                f"RH {condition['relative_humidity_pct']:.0f}%"
            )

        def _update_scene(self) -> None:
            minute = int(model.minutes[self.frame])
            self.time_label.setText(f"T+{minute:03d} min  ·  frame {self.frame + 1}")
            if model.time_origin is not None:
                self.time_label.setText(
                    f"{model.clock_label(minute)}  ·  T+{minute:03d}  ·  frame {self.frame + 1}"
                )
            self._update_vehicle_table()
            if self.config.playback.event_autoselect and model.events:
                event_minutes = np.fromiter(
                    (event.minute for event in model.events),
                    dtype=np.int32,
                )
                event_row = int(np.searchsorted(event_minutes, minute, side="right") - 1)
                if event_row >= 0:
                    self.event_table.selectRow(event_row)
                    item = self.event_table.item(event_row, 0)
                    if item is not None:
                        self.event_table.scrollToItem(item)
            if self.stack.currentIndex() == 0:
                self.map_canvas.update_scene(
                    self.frame,
                    self.config,
                    self.selected_resource,
                )
            else:
                self.terrain_canvas.update_scene(
                    self.frame,
                    self.config,
                    self.selected_resource,
                )

        def _update_vehicle_table(self) -> None:
            for row, resource in enumerate(model.resources(self.frame)):
                endurance = resource["endurance_remaining_min"]
                values = (
                    resource["id"],
                    resource["status_name"],
                    resource["task_name"],
                    f"{resource['payload_fraction']:.0%}",
                    "—" if np.isnan(endurance) else f"{endurance:.0f} min",
                    resource["service_site"] or resource["current_site"] or "—",
                )
                for column, value in enumerate(values):
                    item = self.vehicle_table.item(row, column)
                    if item is None:
                        item = QTableWidgetItem()
                        self.vehicle_table.setItem(row, column, item)
                    item.setText(str(value))
            if self.selected_resource is not None:
                selected = model.resource(self.frame, self.selected_resource)
                self.vehicle_detail.setText(
                    f"{selected['id']} — {selected['status_name']}; "
                    f"task {selected['task_name']}; ETA {selected['eta_min']} min; "
                    f"target ({selected['target_x']:.1f}, {selected['target_y']:.1f})"
                )

        def _export_frame(self) -> None:
            destination, _ = QFileDialog.getSaveFileName(
                self,
                "Export current frame",
                f"aeolus-{self.frame:04d}.png",
                "PNG image (*.png);;TIFF image (*.tif *.tiff)",
            )
            if not destination:
                return
            try:
                render = render_frame_3d if self.stack.currentIndex() == 1 else render_frame_2d
                render(
                    replay,
                    destination,
                    frame=self.frame,
                    viewer_config=self.config,
                    selected_resource=self.selected_resource,
                )
                self.statusBar().showMessage(f"Exported {destination}", 5000)
            except Exception as exc:
                QMessageBox.critical(self, "Export failed", str(exc))

        def _save_config(self) -> None:
            destination, _ = QFileDialog.getSaveFileName(
                self,
                "Save view configuration",
                "viewer.yaml",
                "YAML (*.yaml *.yml)",
            )
            if not destination:
                return
            Path(destination).write_text(
                yaml.safe_dump(self.config.as_dict(), sort_keys=False),
                encoding="utf-8",
            )
            self.statusBar().showMessage(f"Saved {destination}", 5000)

    return ViewerWindow()
