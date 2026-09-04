#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ============================================================
# Configuration
# ============================================================

class Config:

    OUTPUT_JSON = "room_graph.json"
    OUTPUT_IMAGE = "room_graph.png"
    OUTPUT_MATRIX = "matrixclose.parquet"

    SPACE_TYPES = {"IfcSpace"}
    DOOR_TYPES = {"IfcDoor"}
    WINDOW_TYPES = {"IfcWindow"}
    OPENING_TYPES = {"IfcOpeningElement"}

    WALL_TYPES = {
        "IfcWall",
        "IfcWallStandardCase",
    }

    PROXY_TYPES = {
        "IfcBuildingElementProxy",
    }

    # Opening must be this close to BOTH room boundaries.
    BOUNDARY_TOLERANCE_PX = 12.0

    # In this particular catalog, doors appear to be encoded
    # as IfcOpeningElement rather than IfcDoor.
    DOOR_MIN_WIDTH = 60.0
    DOOR_MAX_WIDTH = 150.0

    DOOR_MIN_THICKNESS = 3.0
    DOOR_MAX_THICKNESS = 20.0

    MIN_OPENING_FRACTION = 1e-12
    DEFAULT_DISTANCE = 1.0

    NODE_SIZE = 500
    EDGE_WIDTH = 2.0
    IMAGE_DPI = 200

    # Tolerances used for physical-opening deduplication.
    DEDUP_POSITION_TOLERANCE = 1.0
    DEDUP_SIZE_TOLERANCE = 1.0


# ============================================================
# Geometry
# ============================================================

@dataclass
class Point2D:

    x: float
    y: float

    def distance(
        self,
        other: "Point2D",
    ) -> float:

        return math.hypot(
            self.x - other.x,
            self.y - other.y,
        )


@dataclass
class BoundingBox:

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:

        return self.max_x - self.min_x

    @property
    def height(self) -> float:

        return self.max_y - self.min_y

    @property
    def center(self) -> Point2D:

        return Point2D(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )


class Geometry:

    @staticmethod
    def centroid(
        points: list[Point2D],
    ) -> Point2D:

        if not points:
            return Point2D(
                0.0,
                0.0,
            )

        area = Geometry.polygon_area(
            points
        )

        if abs(area) < 1e-12:

            x = (
                sum(p.x for p in points)
                / len(points)
            )

            y = (
                sum(p.y for p in points)
                / len(points)
            )

            return Point2D(
                x,
                y,
            )

        cx = 0.0
        cy = 0.0

        for i in range(len(points)):

            p1 = points[i]
            p2 = points[
                (i + 1) % len(points)
            ]

            cross = (
                p1.x * p2.y
                - p2.x * p1.y
            )

            cx += (
                p1.x + p2.x
            ) * cross

            cy += (
                p1.y + p2.y
            ) * cross

        factor = (
            1.0 / (6.0 * area)
        )

        return Point2D(
            cx * factor,
            cy * factor,
        )

    @staticmethod
    def polygon_area(
        points: list[Point2D],
    ) -> float:

        if len(points) < 3:
            return 0.0

        value = 0.0

        for i in range(len(points)):

            p1 = points[i]
            p2 = points[
                (i + 1) % len(points)
            ]

            value += (
                p1.x * p2.y
                - p2.x * p1.y
            )

        return abs(value) / 2.0

    @staticmethod
    def point_segment_distance(
        point: Point2D,
        a: Point2D,
        b: Point2D,
    ) -> float:

        dx = b.x - a.x
        dy = b.y - a.y

        if (
            dx == 0.0
            and dy == 0.0
        ):
            return point.distance(a)

        t = (
            (point.x - a.x) * dx
            + (point.y - a.y) * dy
        ) / (
            dx * dx
            + dy * dy
        )

        t = max(
            0.0,
            min(1.0, t),
        )

        projection = Point2D(
            a.x + t * dx,
            a.y + t * dy,
        )

        return point.distance(
            projection
        )

    @staticmethod
    def distance_to_polygon(
        point: Point2D,
        polygon: list[Point2D],
    ) -> float:

        if not polygon:
            return float("inf")

        if len(polygon) == 1:

            return point.distance(
                polygon[0]
            )

        result = float("inf")

        for i in range(len(polygon)):

            a = polygon[i]
            b = polygon[
                (i + 1) % len(polygon)
            ]

            result = min(
                result,
                Geometry.point_segment_distance(
                    point,
                    a,
                    b,
                ),
            )

        return result


# ============================================================
# IFC object
# ============================================================

@dataclass
class IFCObject:

    id: str
    ifc_type: str

    name: str = ""

    geometry: list[Point2D] = field(
        default_factory=list
    )

    width_px: float = 0.0
    height_px: float = 0.0
    length_px: float = 0.0

    width_m: Optional[float] = None
    height_m: Optional[float] = None
    length_m: Optional[float] = None

    area_px2: float = 0.0
    area_m2: Optional[float] = None

    confidence: float = 0.0

    properties: dict[str, Any] = field(
        default_factory=dict
    )

    references: list[str] = field(
        default_factory=list
    )

    referenced_by: list[str] = field(
        default_factory=list
    )

    bbox: Optional[BoundingBox] = None

    def calculate_bbox(self) -> None:

        if not self.geometry:

            self.bbox = None
            return

        xs = [
            p.x
            for p in self.geometry
        ]

        ys = [
            p.y
            for p in self.geometry
        ]

        self.bbox = BoundingBox(
            min_x=min(xs),
            min_y=min(ys),
            max_x=max(xs),
            max_y=max(ys),
        )

    def is_space(self) -> bool:

        return (
            self.ifc_type
            in Config.SPACE_TYPES
        )

    def is_door(self) -> bool:

        return (
            self.ifc_type
            in Config.DOOR_TYPES
        )

    def is_window(self) -> bool:

        return (
            self.ifc_type
            in Config.WINDOW_TYPES
        )

    def is_opening(self) -> bool:

        return (
            self.ifc_type
            in Config.OPENING_TYPES
        )

    def is_proxy(self) -> bool:

        return (
            self.ifc_type
            in Config.PROXY_TYPES
        )

    def is_gap(self) -> bool:

        if not self.is_proxy():
            return False

        text = str(
            self.properties.get(
                "object_type",
                "",
            )
        ).upper()

        return (
            "GAP" in text
            or "OPEN" in text
        )


# ============================================================
# JSON reader
# ============================================================

class IFCJsonReader:

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    def _geometry(
        self,
        raw: dict[str, Any],
    ) -> list[Point2D]:

        result = []

        geometry = raw.get(
            "geometry",
            [],
        )

        if not isinstance(
            geometry,
            list,
        ):
            return result

        for item in geometry:

            if not isinstance(
                item,
                dict,
            ):
                continue

            try:

                result.append(
                    Point2D(
                        x=float(
                            item["x"]
                        ),
                        y=float(
                            item["y"]
                        ),
                    )
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):

                continue

        return result

    def _objects_from_container(
        self,
        container: Any,
    ) -> list[dict[str, Any]]:

        if isinstance(
            container,
            list,
        ):

            return [
                item
                for item in container
                if isinstance(
                    item,
                    dict,
                )
            ]

        if not isinstance(
            container,
            dict,
        ):

            return []

        for key in (
            "objects",
            "ifc_objects",
            "catalog",
            "items",
        ):

            value = container.get(
                key
            )

            if isinstance(
                value,
                list,
            ):

                return [
                    item
                    for item in value
                    if isinstance(
                        item,
                        dict,
                    )
                ]

        return []

    def read(
        self,
        filename: str | Path,
    ) -> list[IFCObject]:

        filename = Path(filename)

        with filename.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        raw_objects = (
            self._objects_from_container(
                data
            )
        )

        if (
            not raw_objects
            and isinstance(data, dict)
        ):

            for value in data.values():

                raw_objects = (
                    self._objects_from_container(
                        value
                    )
                )

                if raw_objects:
                    break

        objects = []

        for raw in raw_objects:

            ifc_type = str(
                raw.get(
                    "ifc_type",
                    raw.get(
                        "ifc_class",
                        raw.get(
                            "type",
                            "",
                        ),
                    ),
                )
            )

            object_id = str(
                raw.get(
                    "id",
                    raw.get(
                        "global_id",
                        "",
                    ),
                )
            )

            if not object_id:
                continue

            obj = IFCObject(
                id=object_id,
                ifc_type=ifc_type,

                name=str(
                    raw.get(
                        "name",
                        "",
                    )
                ),

                geometry=self._geometry(
                    raw
                ),

                width_px=self._number(
                    raw.get(
                        "width_px"
                    )
                ),

                height_px=self._number(
                    raw.get(
                        "height_px"
                    )
                ),

                length_px=self._number(
                    raw.get(
                        "length_px"
                    )
                ),

                width_m=(
                    self._number(
                        raw["width_m"]
                    )
                    if raw.get(
                        "width_m"
                    ) is not None
                    else None
                ),

                height_m=(
                    self._number(
                        raw["height_m"]
                    )
                    if raw.get(
                        "height_m"
                    ) is not None
                    else None
                ),

                length_m=(
                    self._number(
                        raw["length_m"]
                    )
                    if raw.get(
                        "length_m"
                    ) is not None
                    else None
                ),

                area_px2=self._number(
                    raw.get(
                        "area_px2"
                    )
                ),

                area_m2=(
                    self._number(
                        raw["area_m2"]
                    )
                    if raw.get(
                        "area_m2"
                    ) is not None
                    else None
                ),

                confidence=self._number(
                    raw.get(
                        "confidence"
                    )
                ),

                properties=(
                    raw.get(
                        "properties",
                        {},
                    )
                    if isinstance(
                        raw.get(
                            "properties",
                            {},
                        ),
                        dict,
                    )
                    else {}
                ),

                references=(
                    raw.get(
                        "references",
                        [],
                    )
                    if isinstance(
                        raw.get(
                            "references",
                            [],
                        ),
                        list,
                    )
                    else []
                ),

                referenced_by=(
                    raw.get(
                        "referenced_by",
                        [],
                    )
                    if isinstance(
                        raw.get(
                            "referenced_by",
                            [],
                        ),
                        list,
                    )
                    else []
                ),
            )

            obj.calculate_bbox()

            objects.append(obj)

        return objects


# ============================================================
# Room
# ============================================================

@dataclass
class Room:

    object: IFCObject

    @property
    def id(self) -> str:

        return self.object.id

    @property
    def name(self) -> str:

        return self.object.name

    @property
    def geometry(
        self,
    ) -> list[Point2D]:

        return self.object.geometry

    @property
    def center(self) -> Point2D:

        if self.geometry:

            return Geometry.centroid(
                self.geometry
            )

        if self.object.bbox:

            return self.object.bbox.center

        return Point2D(
            0.0,
            0.0,
        )

    @property
    def area(self) -> float:

        if self.object.area_px2 > 0:

            return self.object.area_px2

        return Geometry.polygon_area(
            self.geometry
        )


# ============================================================
# Opening
# ============================================================

class OpeningType(Enum):

    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"
    GAP = "open_gap"
    UNKNOWN = "unknown"


@dataclass
class Opening:

    object: IFCObject
    opening_type: OpeningType

    @property
    def id(self) -> str:

        return self.object.id

    @property
    def center(self) -> Point2D:

        if self.object.geometry:

            return Geometry.centroid(
                self.object.geometry
            )

        if self.object.bbox:

            return self.object.bbox.center

        return Point2D(
            0.0,
            0.0,
        )

    @property
    def width(self) -> float:

        if self.object.width_px > 0:

            return self.object.width_px

        if (
            self.object.width_m is not None
            and self.object.width_m > 0
        ):

            return self.object.width_m

        if self.object.bbox:

            return max(
                self.object.bbox.width,
                self.object.bbox.height,
            )

        return 0.0

    @property
    def height(self) -> float:

        if self.object.height_px > 0:

            return self.object.height_px

        if (
            self.object.height_m is not None
            and self.object.height_m > 0
        ):

            return self.object.height_m

        if self.object.bbox:

            return min(
                self.object.bbox.width,
                self.object.bbox.height,
            )

        return 0.0

    @property
    def area(self) -> float:

        if self.object.area_px2 > 0:

            return self.object.area_px2

        return (
            self.width
            * self.height
        )


# ============================================================
# Opening classifier
# ============================================================

class OpeningClassifier:

    def classify(
        self,
        obj: IFCObject,
    ) -> Optional[Opening]:

        if obj.is_door():

            return Opening(
                object=obj,
                opening_type=OpeningType.DOOR,
            )

        if obj.is_window():

            return Opening(
                object=obj,
                opening_type=OpeningType.WINDOW,
            )

        if obj.is_opening():

            return Opening(
                object=obj,
                opening_type=OpeningType.OPENING,
            )

        if obj.is_gap():

            return Opening(
                object=obj,
                opening_type=OpeningType.GAP,
            )

        return None


# ============================================================
# Opening association
# ============================================================

@dataclass
class OpeningAssociation:

    opening: Opening
    rooms: list[Room]
    distances: list[float]

    @property
    def valid(self) -> bool:

        return len(
            self.rooms
        ) == 2


class OpeningRoomAssociator:

    def __init__(
        self,
        tolerance_px: float =
            Config.BOUNDARY_TOLERANCE_PX,
    ):

        self.tolerance_px = (
            tolerance_px
        )

    def associate(
        self,
        opening: Opening,
        rooms: list[Room],
    ) -> OpeningAssociation:

        candidates = []

        for room in rooms:

            if not room.geometry:
                continue

            distance = (
                Geometry.distance_to_polygon(
                    opening.center,
                    room.geometry,
                )
            )

            candidates.append(
                (
                    distance,
                    room,
                )
            )

        candidates.sort(
            key=lambda value: value[0]
        )

        if not candidates:

            return OpeningAssociation(
                opening=opening,
                rooms=[],
                distances=[],
            )

        # IMPORTANT:
        #
        # Do NOT use the two nearest rooms
        # as a fallback.
        #
        # The opening must really be close
        # to both room boundaries.

        close = [
            item
            for item in candidates
            if item[0]
            <= self.tolerance_px
        ]

        if len(close) < 2:

            return OpeningAssociation(
                opening=opening,
                rooms=[],
                distances=[],
            )

        selected = close[:2]

        return OpeningAssociation(
            opening=opening,
            rooms=[
                item[1]
                for item in selected
            ],
            distances=[
                item[0]
                for item in selected
            ],
        )


# ============================================================
# Solid angle
# ============================================================

class SolidAngleCalculator:

    @staticmethod
    def rectangular_solid_angle(
        width: float,
        height: float,
        distance: float,
    ) -> float:

        if (
            width <= 0
            or height <= 0
            or distance <= 0
        ):

            return 0.0

        denominator = (
            distance
            * math.sqrt(
                distance * distance
                + width * width
                + height * height
            )
        )

        value = (
            width * height
        ) / denominator

        return (
            4.0
            * math.atan(value)
        )

    @staticmethod
    def fraction(
        solid_angle: float,
    ) -> float:

        return (
            solid_angle
            / (4.0 * math.pi)
        )


# ============================================================
# Edge weight
# ============================================================

class EdgeWeightCalculator:

    def calculate(
        self,
        room_a: Room,
        room_b: Room,
        opening: Opening,
    ) -> dict[str, float]:

        distance = (
            room_a.center.distance(
                room_b.center
            )
        )

        if distance <= 0:

            distance = (
                Config.DEFAULT_DISTANCE
            )

        solid_angle = (
            SolidAngleCalculator
            .rectangular_solid_angle(
                opening.width,
                opening.height,
                distance,
            )
        )

        fraction = (
            SolidAngleCalculator
            .fraction(
                solid_angle
            )
        )

        fraction = max(
            fraction,
            Config.MIN_OPENING_FRACTION,
        )

        weight = (
            fraction
            / (distance * distance)
        )

        return {
            "distance": distance,
            "distance_squared":
                distance * distance,
            "solid_angle":
                solid_angle,
            "opening_fraction":
                fraction,
            "weight":
                weight,
        }


# ============================================================
# Graph node
# ============================================================

@dataclass
class GraphNode:

    id: str
    name: str
    area: float
    x: float
    y: float

    def as_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "id": self.id,
            "name": self.name,
            "area": self.area,
            "x": self.x,
            "y": self.y,
        }


# ============================================================
# Graph edge
# ============================================================

@dataclass
class GraphEdge:

    source: str
    target: str

    connection_type: str
    closable: bool

    opening_id: str

    distance: float
    distance_squared: float

    solid_angle: float
    opening_fraction: float
    weight: float

    def as_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "source": self.source,
            "target": self.target,
            "connection_type":
                self.connection_type,
            "closable": self.closable,
            "opening_id":
                self.opening_id,
            "distance":
                self.distance,
            "distance_squared":
                self.distance_squared,
            "solid_angle":
                self.solid_angle,
            "opening_fraction":
                self.opening_fraction,
            "weight":
                self.weight,
        }


# ============================================================
# Room graph
# ============================================================

class RoomGraph:

    def __init__(self):

        self.nodes: dict[
            str,
            GraphNode,
        ] = {}

        self.edges: list[
            GraphEdge
        ] = []

    def add_room(
        self,
        room: Room,
    ) -> None:

        center = room.center

        self.nodes[room.id] = GraphNode(
            id=room.id,
            name=room.name,
            area=room.area,
            x=center.x,
            y=center.y,
        )

    def add_edge(
        self,
        edge: GraphEdge,
    ) -> None:

        self.edges.append(
            edge
        )

    def closable_edges(
        self,
    ) -> list[GraphEdge]:

        return [
            edge
            for edge in self.edges
            if edge.closable
        ]

    def open_edges(
        self,
    ) -> list[GraphEdge]:

        return [
            edge
            for edge in self.edges
            if not edge.closable
        ]

    def as_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "nodes": [
                node.as_dict()
                for node in self.nodes.values()
            ],
            "edges": [
                edge.as_dict()
                for edge in self.edges
            ],
        }


# ============================================================
# Door inference
# ============================================================

class DoorInference:

    @staticmethod
    def is_door(
        opening: Opening,
        association: OpeningAssociation,
    ) -> bool:

        # Explicit IfcDoor.
        if (
            opening.opening_type
            == OpeningType.DOOR
        ):

            return True

        # Only infer doors from
        # IfcOpeningElement.
        if (
            opening.opening_type
            != OpeningType.OPENING
        ):

            return False

        if not association.valid:

            return False

        width = opening.width
        height = opening.height

        long_dimension = max(
            width,
            height,
        )

        short_dimension = min(
            width,
            height,
        )

        dimension_ok = (
            Config.DOOR_MIN_WIDTH
            <= long_dimension
            <= Config.DOOR_MAX_WIDTH
            and
            Config.DOOR_MIN_THICKNESS
            <= short_dimension
            <= Config.DOOR_MAX_THICKNESS
        )

        boundary_ok = (
            association.distances[0]
            <= Config.BOUNDARY_TOLERANCE_PX
            and
            association.distances[1]
            <= Config.BOUNDARY_TOLERANCE_PX
        )

        return (
            dimension_ok
            and boundary_ok
        )


# ============================================================
# Graph builder
# ============================================================

class RoomGraphBuilder:

    def __init__(self):

        self.classifier = (
            OpeningClassifier()
        )

        self.associator = (
            OpeningRoomAssociator()
        )

        self.weight_calculator = (
            EdgeWeightCalculator()
        )

    # --------------------------------------------------------
    # Physical opening key
    # --------------------------------------------------------

    @staticmethod
    def _physical_key(
        opening: Opening,
        association: OpeningAssociation,
    ) -> tuple:

        if not association.valid:

            return ()

        room_ids = tuple(
            sorted(
                (
                    association.rooms[0].id,
                    association.rooms[1].id,
                )
            )
        )

        center = opening.center

        position_tol = (
            Config.DEDUP_POSITION_TOLERANCE
        )

        size_tol = (
            Config.DEDUP_SIZE_TOLERANCE
        )

        return (
            room_ids,

            round(
                center.x
                / position_tol
            ),

            round(
                center.y
                / position_tol
            ),

            round(
                opening.width
                / size_tol
            ),

            round(
                opening.height
                / size_tol
            ),
        )

    # --------------------------------------------------------
    # Deduplicate physical openings
    # --------------------------------------------------------

    def _deduplicate_openings(
        self,
        associated: list[
            tuple[
                Opening,
                OpeningAssociation,
            ]
        ],
    ) -> list[
        tuple[
            Opening,
            OpeningAssociation,
        ]
    ]:

        unique: dict[
            tuple,
            tuple[
                Opening,
                OpeningAssociation,
            ],
        ] = {}

        duplicate_count = 0

        for opening, association in associated:

            key = self._physical_key(
                opening,
                association,
            )

            if not key:

                continue

            if key in unique:

                existing_opening = (
                    unique[key][0]
                )

                duplicate_count += 1

                print(
                    f"  Duplicate opening "
                    f"{opening.id} "
                    f"-> "
                    f"{existing_opening.id} "
                    f"(same physical opening)"
                )

                continue

            unique[key] = (
                opening,
                association,
            )

        result = list(
            unique.values()
        )

        print(
            f"Unique physical openings: "
            f"{len(result)}"
        )

        print(
            f"Duplicate openings removed: "
            f"{duplicate_count}"
        )

        return result

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    def build(
        self,
        objects: list[IFCObject],
    ) -> RoomGraph:

        rooms = [
            Room(object=obj)
            for obj in objects
            if obj.is_space()
        ]

        graph = RoomGraph()

        for room in rooms:

            graph.add_room(
                room
            )

        openings = []

        for obj in objects:

            opening = (
                self.classifier.classify(
                    obj
                )
            )

            if opening is not None:

                openings.append(
                    opening
                )

        print(
            f"Potential openings: "
            f"{len(openings)}"
        )

        # ----------------------------------------------------
        # First associate openings with rooms.
        # ----------------------------------------------------

        associated: list[
            tuple[
                Opening,
                OpeningAssociation,
            ]
        ] = []

        ignored = 0

        for opening in openings:

            # Windows do not represent
            # room passage.

            if (
                opening.opening_type
                == OpeningType.WINDOW
            ):

                continue

            association = (
                self.associator.associate(
                    opening,
                    rooms,
                )
            )

            if not association.valid:

                print(
                    f"  Ignored opening "
                    f"{opening.id}: "
                    f"could not associate "
                    f"with two rooms."
                )

                ignored += 1

                continue

            associated.append(
                (
                    opening,
                    association,
                )
            )

        print(
            f"Associated openings: "
            f"{len(associated)}"
        )

        # ----------------------------------------------------
        # Deduplicate physical openings.
        # ----------------------------------------------------

        unique = (
            self._deduplicate_openings(
                associated
            )
        )

        # ----------------------------------------------------
        # Build edges.
        # ----------------------------------------------------

        door_count = 0
        open_count = 0

        for opening, association in unique:

            room_a = (
                association.rooms[0]
            )

            room_b = (
                association.rooms[1]
            )

            # ------------------------------------------------
            # Door inference.
            # ------------------------------------------------

            closable = (
                DoorInference.is_door(
                    opening,
                    association,
                )
            )

            if (
                opening.opening_type
                == OpeningType.OPENING
            ):

                width = opening.width
                height = opening.height

                long_dimension = max(
                    width,
                    height,
                )

                short_dimension = min(
                    width,
                    height,
                )

                dimension_ok = (
                    Config.DOOR_MIN_WIDTH
                    <= long_dimension
                    <= Config.DOOR_MAX_WIDTH
                    and
                    Config.DOOR_MIN_THICKNESS
                    <= short_dimension
                    <= Config.DOOR_MAX_THICKNESS
                )

                boundary_ok = (
                    association.distances[0]
                    <= Config.BOUNDARY_TOLERANCE_PX
                    and
                    association.distances[1]
                    <= Config.BOUNDARY_TOLERANCE_PX
                )

                print(
                    f"DOOR TEST "
                    f"{opening.id}: "
                    f"size="
                    f"{width:.3f}x"
                    f"{height:.3f} "
                    f"boundary="
                    f"{association.distances[0]:.3f},"
                    f"{association.distances[1]:.3f} "
                    f"dimension="
                    f"{'YES' if dimension_ok else 'NO'} "
                    f"boundary="
                    f"{'YES' if boundary_ok else 'NO'} "
                    f"DOOR="
                    f"{'YES' if closable else 'NO'}"
                )

            edge_weight = (
                self.weight_calculator.calculate(
                    room_a,
                    room_b,
                    opening,
                )
            )

            if closable:

                connection_type = "door"
                door_count += 1

            elif (
                opening.opening_type
                == OpeningType.GAP
            ):

                connection_type = "open_gap"
                open_count += 1

            else:

                connection_type = "opening"
                open_count += 1

            edge = GraphEdge(
                source=room_a.id,
                target=room_b.id,

                connection_type=
                    connection_type,

                closable=closable,

                opening_id=
                    opening.id,

                distance=
                    edge_weight["distance"],

                distance_squared=
                    edge_weight[
                        "distance_squared"
                    ],

                solid_angle=
                    edge_weight[
                        "solid_angle"
                    ],

                opening_fraction=
                    edge_weight[
                        "opening_fraction"
                    ],

                weight=
                    edge_weight["weight"],
            )

            graph.add_edge(
                edge
            )

        print()
        print("Classification:")

        print(
            f"  Doors:              "
            f"{door_count}"
        )

        print(
            f"  Permanent openings: "
            f"{open_count}"
        )

        print(
            f"  Ignored:             "
            f"{ignored}"
        )

        return graph


# ============================================================
# Graph analysis
# ============================================================

class GraphAnalyzer:

    def __init__(
        self,
        graph: RoomGraph,
    ):

        self.graph = graph

    def components(
        self,
        doors_closed: bool = False,
    ) -> list[list[str]]:

        adjacency = {
            room_id: set()
            for room_id
            in self.graph.nodes
        }

        for edge in self.graph.edges:

            if (
                doors_closed
                and edge.closable
            ):

                continue

            adjacency[
                edge.source
            ].add(
                edge.target
            )

            adjacency[
                edge.target
            ].add(
                edge.source
            )

        result = []
        visited = set()

        for start in adjacency:

            if start in visited:

                continue

            component = []
            stack = [start]

            while stack:

                current = stack.pop()

                if current in visited:

                    continue

                visited.add(
                    current
                )

                component.append(
                    current
                )

                stack.extend(
                    adjacency[current]
                    - visited
                )

            result.append(
                sorted(component)
            )

        return result

    def permanently_connected_components(
        self,
    ) -> list[list[str]]:

        return self.components(
            doors_closed=True
        )


# ============================================================
# Closable matrix
# ============================================================

class ClosableGraphMatrix:

    def __init__(
        self,
        graph: RoomGraph,
    ):

        self.graph = graph

    def write(
        self,
        filename: str | Path,
    ) -> None:

        import pyarrow as pa
        import pyarrow.parquet as pq

        room_ids = list(
            self.graph.nodes.keys()
        )

        index = {
            room_id: i
            for i, room_id
            in enumerate(room_ids)
        }

        matrix = {
            room_id: [0.0] * len(room_ids)
            for room_id in room_ids
        }

        for edge in (
            self.graph.closable_edges()
        ):

            i = index[
                edge.source
            ]

            j = index[
                edge.target
            ]

            matrix[
                edge.source
            ][j] += edge.weight

            matrix[
                edge.target
            ][i] += edge.weight

        data = {
            "room_id": room_ids
        }

        for room_id in room_ids:

            data[room_id] = matrix[
                room_id
            ]

        table = pa.table(
            data
        )

        pq.write_table(
            table,
            filename,
        )


# ============================================================
# JSON writer
# ============================================================

class RoomGraphJsonWriter:

    def write(
        self,
        filename: str | Path,
        graph: RoomGraph,
        analyzer: GraphAnalyzer,
    ) -> None:

        data = graph.as_dict()

        data["analysis"] = {
            "components_doors_open":
                analyzer.components(
                    doors_closed=False
                ),

            "components_doors_closed":
                analyzer.permanently_connected_components(),
        }

        data["matrix"] = {
            "file":
                Config.OUTPUT_MATRIX,

            "description":
                "Symmetric weighted adjacency "
                "matrix of closable door "
                "connections.",

            "weight_formula":
                "opening_fraction / distance^2",

            "opening_fraction":
                "solid_angle / (4*pi)",

            "nonzero_edges":
                len(
                    graph.closable_edges()
                ),
        }

        with Path(filename).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )


# ============================================================
# Visualization
# ============================================================

class GraphVisualizer:

    def visualize(
        self,
        graph: RoomGraph,
        filename: str | Path,
    ) -> None:

        import matplotlib.pyplot as plt

        plt.figure(
            figsize=(14, 10)
        )

        # ----------------------------------------------------
        # Edges.
        # ----------------------------------------------------

        for edge in graph.edges:

            source = graph.nodes[
                edge.source
            ]

            target = graph.nodes[
                edge.target
            ]

            linestyle = (
                "--"
                if edge.closable
                else "-"
            )

            plt.plot(
                [
                    source.x,
                    target.x,
                ],
                [
                    source.y,
                    target.y,
                ],
                linestyle=linestyle,
                linewidth=Config.EDGE_WIDTH,
            )

            x = (
                source.x
                + target.x
            ) / 2.0

            y = (
                source.y
                + target.y
            ) / 2.0

            plt.text(
                x,
                y,
                f"{edge.connection_type}\n"
                f"{edge.weight:.2e}",
                fontsize=7,
            )

        # ----------------------------------------------------
        # Nodes.
        # ----------------------------------------------------

        xs = [
            node.x
            for node
            in graph.nodes.values()
        ]

        ys = [
            node.y
            for node
            in graph.nodes.values()
        ]

        plt.scatter(
            xs,
            ys,
            s=Config.NODE_SIZE,
        )

        for node in graph.nodes.values():

            label = (
                node.name
                if node.name
                else node.id
            )

            plt.text(
                node.x,
                node.y,
                label,
                fontsize=8,
            )

        plt.axis(
            "equal"
        )

        plt.xlabel(
            "X"
        )

        plt.ylabel(
            "Y"
        )

        plt.title(
            "Room Connectivity Graph"
        )

        plt.savefig(
            filename,
            dpi=Config.IMAGE_DPI,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# Application
# ============================================================

class RoomGraphApplication:

    def __init__(
        self,
        input_file: str,
        output_json: str,
        output_image: str,
        output_matrix: str,
    ):

        self.input_file = input_file
        self.output_json = output_json
        self.output_image = output_image
        self.output_matrix = output_matrix

    def run(self) -> None:

        reader = IFCJsonReader()

        objects = reader.read(
            self.input_file
        )

        print(
            f"Objects: {len(objects)}"
        )

        spaces = sum(
            obj.is_space()
            for obj in objects
        )

        doors = sum(
            obj.is_door()
            for obj in objects
        )

        windows = sum(
            obj.is_window()
            for obj in objects
        )

        openings = sum(
            obj.is_opening()
            for obj in objects
        )

        gaps = sum(
            obj.is_gap()
            for obj in objects
        )

        print()
        print(
            "IFC object statistics:"
        )

        print(
            f"  Spaces:   {spaces}"
        )

        print(
            f"  Doors:    {doors}"
        )

        print(
            f"  Windows:  {windows}"
        )

        print(
            f"  Openings: {openings}"
        )

        print(
            f"  Gaps:     {gaps}"
        )

        builder = RoomGraphBuilder()

        graph = builder.build(
            objects
        )

        print()
        print(
            "Graph:"
        )

        print(
            f"  Nodes: "
            f"{len(graph.nodes)}"
        )

        print(
            f"  Edges: "
            f"{len(graph.edges)}"
        )

        print(
            f"  Closable edges: "
            f"{len(graph.closable_edges())}"
        )

        print(
            f"  Open edges: "
            f"{len(graph.open_edges())}"
        )

        print()
        print(
            "Edges:"
        )

        for edge in graph.edges:

            state = (
                "CLOSABLE"
                if edge.closable
                else "OPEN"
            )

            print(
                f"  "
                f"{edge.source} <-> "
                f"{edge.target} | "
                f"{edge.connection_type} | "
                f"{state} | "
                f"d={edge.distance:.3f} | "
                f"f={edge.opening_fraction:.8g} | "
                f"w={edge.weight:.8g}"
            )

        analyzer = GraphAnalyzer(
            graph
        )

        print()
        print(
            "Connected components "
            "(doors open):"
        )

        for component in (
            analyzer.components(
                doors_closed=False
            )
        ):

            print(
                f"  {component}"
            )

        print()
        print(
            "Permanently connected "
            "components (doors closed):"
        )

        for component in (
            analyzer.permanently_connected_components()
        ):

            print(
                f"  {component}"
            )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        RoomGraphJsonWriter().write(
            self.output_json,
            graph,
            analyzer,
        )

        print()
        print(
            f"Wrote {self.output_json}"
        )

        # ----------------------------------------------------
        # Matrix
        # ----------------------------------------------------

        closable = (
            graph.closable_edges()
        )

        if closable:

            print()
            print(
                "Closable edges for "
                "matrixclose.parquet:"
            )

            for edge in closable:

                print(
                    f"  "
                    f"{edge.source} <-> "
                    f"{edge.target} "
                    f"weight="
                    f"{edge.weight:.8g}"
                )

        else:

            print()
            print(
                "*** NO CLOSABLE EDGES ***"
            )

        ClosableGraphMatrix(
            graph
        ).write(
            self.output_matrix
        )

        print()
        print(
            f"Wrote {self.output_matrix}"
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        GraphVisualizer().visualize(
            graph,
            self.output_image,
        )

        print(
            f"Wrote {self.output_image}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build a room connectivity "
            "graph from IFC JSON."
        )
    )

    parser.add_argument(
        "input",
        help="Input catalog JSON file",
    )

    parser.add_argument(
        "--output",
        default=Config.OUTPUT_JSON,
        help="Output graph JSON",
    )

    parser.add_argument(
        "--visualization",
        default=Config.OUTPUT_IMAGE,
        help="Output graph PNG",
    )

    parser.add_argument(
        "--matrix",
        default=Config.OUTPUT_MATRIX,
        help="Output closable matrix Parquet",
    )

    args = parser.parse_args()

    application = RoomGraphApplication(
        input_file=args.input,
        output_json=args.output,
        output_image=args.visualization,
        output_matrix=args.matrix,
    )

    application.run()


if __name__ == "__main__":

    main()
