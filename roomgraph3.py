#!/usr/bin/env python3

"""
roomgrapher.py

Build a weighted room connectivity graph from an IFC-style JSON file.

Input:
    IFC semantic JSON produced by the floor-plan analyzer.

Graph nodes:
    IfcSpace

Graph connections:
    IfcDoor
        -> closable connection

    IfcOpeningElement
        -> permanently open connection

    IfcBuildingElementProxy with "gap"/"open"
        -> permanently open connection

    IfcWindow
        -> ignored as room-to-room passage

Weight:

    opening_fraction = solid_angle / (4*pi)

    weight = opening_fraction / distance^2

where distance is the distance between the centers of the
two connected rooms.

Outputs:

    room_graph.json
    room_graph.png
    matrixclose.parquet


Important:
    matrixclose.parquet contains ONLY closable connections,
    i.e. doors.

The matrix is symmetric:

    M[A,B] = M[B,A]

If several doors connect the same two rooms, their weights
are added.
"""

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

    SPACE_TYPES = {
        "IfcSpace",
    }

    DOOR_TYPES = {
        "IfcDoor",
    }

    WINDOW_TYPES = {
        "IfcWindow",
    }

    OPENING_TYPES = {
        "IfcOpeningElement",
    }

    PROXY_TYPES = {
        "IfcBuildingElementProxy",
    }

    BOUNDARY_TOLERANCE_PX = 20.0

    DEFAULT_DISTANCE = 1.0

    MIN_OPENING_FRACTION = 1e-12

    EDGE_WIDTH = 2.0

    NODE_SIZE = 500

    IMAGE_DPI = 200


# ============================================================
# Geometry
# ============================================================

@dataclass(frozen=True)
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

        return max(
            0.0,
            self.max_x - self.min_x,
        )

    @property
    def height(self) -> float:

        return max(
            0.0,
            self.max_y - self.min_y,
        )

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
    ) -> Optional[Point2D]:

        if not points:
            return None

        return Point2D(
            sum(p.x for p in points) / len(points),
            sum(p.y for p in points) / len(points),
        )

    @staticmethod
    def polygon_area(
        points: list[Point2D],
    ) -> float:

        if len(points) < 3:
            return 0.0

        area = 0.0

        for i, p1 in enumerate(points):

            p2 = points[
                (i + 1) % len(points)
            ]

            area += (
                p1.x * p2.y
                - p2.x * p1.y
            )

        return abs(area) / 2.0

    @staticmethod
    def point_segment_distance(
        point: Point2D,
        a: Point2D,
        b: Point2D,
    ) -> float:

        dx = b.x - a.x
        dy = b.y - a.y

        length_squared = (
            dx * dx +
            dy * dy
        )

        if length_squared == 0:

            return point.distance(a)

        t = (
            (point.x - a.x) * dx
            + (point.y - a.y) * dy
        ) / length_squared

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

        minimum = float("inf")

        for i, p1 in enumerate(polygon):

            p2 = polygon[
                (i + 1) % len(polygon)
            ]

            d = (
                Geometry.point_segment_distance(
                    point,
                    p1,
                    p2,
                )
            )

            minimum = min(
                minimum,
                d,
            )

        return minimum


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

        return self.ifc_type in Config.SPACE_TYPES

    def is_door(self) -> bool:

        return self.ifc_type in Config.DOOR_TYPES

    def is_window(self) -> bool:

        return self.ifc_type in Config.WINDOW_TYPES

    def is_opening(self) -> bool:

        return self.ifc_type in Config.OPENING_TYPES

    def is_proxy(self) -> bool:

        return self.ifc_type in Config.PROXY_TYPES

    def is_gap(self) -> bool:

        if not self.is_proxy():
            return False

        values = [
            self.properties.get(
                "object_type",
                ""
            ),
            self.properties.get(
                "type",
                ""
            ),
            self.name,
        ]

        text = " ".join(
            str(v)
            for v in values
        ).upper()

        return (
            "GAP" in text
            or "OPENING" in text
            or "OPEN" in text
        )


# ============================================================
# IFC JSON reader
# ============================================================

class IFCJsonReader:

    def __init__(
        self,
        filename: str | Path,
    ):

        self.filename = Path(
            filename
        )

    def read(
        self,
    ) -> list[IFCObject]:

        print(
            f"Reading IFC JSON: "
            f"{self.filename}"
        )

        with self.filename.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        raw_objects = (
            self._find_objects(data)
        )

        objects = []

        for raw in raw_objects:

            obj = self._create_object(
                raw
            )

            if obj is None:
                continue

            obj.calculate_bbox()

            objects.append(obj)

        return objects

    def _find_objects(
        self,
        data: Any,
    ) -> list[dict[str, Any]]:

        if isinstance(data, list):

            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        if not isinstance(data, dict):

            raise ValueError(
                "Invalid JSON structure."
            )

        # Common catalog keys.
        for key in (
            "objects",
            "ifc_objects",
            "catalog",
            "items",
        ):

            value = data.get(key)

            if isinstance(value, list):

                objects = [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

                if objects:
                    return objects

        # Search recursively.
        result = []

        self._collect_objects(
            data,
            result,
        )

        if result:
            return result

        raise ValueError(
            "No IFC objects found in JSON."
        )

    def _collect_objects(
        self,
        value: Any,
        result: list[dict[str, Any]],
    ) -> None:

        if isinstance(value, dict):

            if (
                "id" in value
                and (
                    "ifc_type" in value
                    or "ifc_class" in value
                    or "type" in value
                )
            ):

                result.append(
                    value
                )

                return

            for child in value.values():

                self._collect_objects(
                    child,
                    result,
                )

        elif isinstance(value, list):

            for child in value:

                self._collect_objects(
                    child,
                    result,
                )

    def _create_object(
        self,
        raw: dict[str, Any],
    ) -> Optional[IFCObject]:

        if "id" not in raw:

            return None

        object_id = str(
            raw["id"]
        )

        # Support all three likely names.
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

        properties = raw.get(
            "properties",
            {},
        )

        if not isinstance(
            properties,
            dict,
        ):

            properties = {}

        geometry = (
            self._read_geometry(
                raw.get(
                    "geometry",
                    [],
                )
            )
        )

        references = (
            self._read_string_list(
                raw.get(
                    "references",
                    [],
                )
            )
        )

        referenced_by = (
            self._read_string_list(
                raw.get(
                    "referenced_by",
                    [],
                )
            )
        )

        return IFCObject(

            id=object_id,

            ifc_type=ifc_type,

            name=str(
                raw.get(
                    "name",
                    "",
                )
            ),

            geometry=geometry,

            width_px=self._number(
                raw.get(
                    "width_px",
                    0,
                )
            ),

            height_px=self._number(
                raw.get(
                    "height_px",
                    0,
                )
            ),

            length_px=self._number(
                raw.get(
                    "length_px",
                    0,
                )
            ),

            width_m=self._optional_number(
                raw.get(
                    "width_m"
                )
            ),

            height_m=self._optional_number(
                raw.get(
                    "height_m"
                )
            ),

            length_m=self._optional_number(
                raw.get(
                    "length_m"
                )
            ),

            area_px2=self._number(
                raw.get(
                    "area_px2",
                    0,
                )
            ),

            area_m2=self._optional_number(
                raw.get(
                    "area_m2"
                )
            ),

            confidence=self._number(
                raw.get(
                    "confidence",
                    0,
                )
            ),

            properties=properties,

            references=references,

            referenced_by=referenced_by,
        )

    @staticmethod
    def _read_geometry(
        value: Any,
    ) -> list[Point2D]:

        if not isinstance(
            value,
            list,
        ):

            return []

        result = []

        for point in value:

            if not isinstance(
                point,
                dict,
            ):
                continue

            if (
                "x" not in point
                or "y" not in point
            ):
                continue

            try:

                result.append(
                    Point2D(
                        float(point["x"]),
                        float(point["y"]),
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        return result

    @staticmethod
    def _read_string_list(
        value: Any,
    ) -> list[str]:

        if isinstance(
            value,
            str,
        ):

            return [value]

        if not isinstance(
            value,
            list,
        ):

            return []

        return [
            str(item)
            for item in value
        ]

    @staticmethod
    def _number(
        value: Any,
    ) -> float:

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    @staticmethod
    def _optional_number(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return None


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

        if self.object.name:
            return self.object.name

        return self.object.id

    @property
    def geometry(self) -> list[Point2D]:

        return self.object.geometry

    @property
    def center(self) -> Point2D:

        center = Geometry.centroid(
            self.geometry
        )

        if center is not None:
            return center

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

    GAP = "gap"

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

        center = Geometry.centroid(
            self.object.geometry
        )

        if center is not None:
            return center

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

        if self.object.width_m is not None:

            return self.object.width_m

        if self.object.bbox:

            return max(
                self.object.bbox.width,
                self.object.bbox.height,
            )

        if len(
            self.object.geometry
        ) >= 2:

            return (
                self.object.geometry[0]
                .distance(
                    self.object.geometry[1]
                )
            )

        return 0.0

    @property
    def height(self) -> float:

        if self.object.height_px > 0:

            return self.object.height_px

        if self.object.height_m is not None:

            return self.object.height_m

        if self.object.bbox:

            return min(
                self.object.bbox.width,
                self.object.bbox.height,
            )

        return 0.0


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
# Opening / room association
# ============================================================

@dataclass
class OpeningAssociation:

    opening: Opening

    rooms: list[Room]

    distances: list[float]

    @property
    def valid(self) -> bool:

        return len(self.rooms) == 2


class OpeningRoomAssociator:
    """
    Associate an opening with two rooms.

    First tries explicit references.

    If that fails, it chooses the two rooms whose boundaries
    are closest to the opening.

    The two closest rooms are NOT rejected simply because
    their distance is greater than a fixed tolerance.
    """

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

        # ----------------------------------------------------
        # Explicit references
        # ----------------------------------------------------

        references = set(
            opening.object.references
            + opening.object.referenced_by
        )

        referenced_rooms = [
            room
            for room in rooms
            if room.id in references
        ]

        if len(
            referenced_rooms
        ) >= 2:

            selected = (
                referenced_rooms[:2]
            )

            return OpeningAssociation(
                opening=opening,
                rooms=selected,
                distances=[
                    Geometry.distance_to_polygon(
                        opening.center,
                        room.geometry,
                    )
                    for room in selected
                ],
            )

        # ----------------------------------------------------
        # Geometric association
        # ----------------------------------------------------

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
            key=lambda item: item[0]
        )

        if len(candidates) < 2:

            return OpeningAssociation(
                opening=opening,
                rooms=[],
                distances=[],
            )

        selected = candidates[:2]

        return OpeningAssociation(
            opening=opening,
            rooms=[
                selected[0][1],
                selected[1][1],
            ],
            distances=[
                selected[0][0],
                selected[1][0],
            ],
        )


# ============================================================
# Solid angle
# ============================================================

class SolidAngleCalculator:
    """
    Solid angle of a rectangle.

    Omega =
        4 atan(
            w*h /
            (
                d * sqrt(
                    d^2 + w^2 + h^2
                )
            )
        )
    """

    @staticmethod
    def rectangular(
        width: float,
        height: float,
        distance: float,
    ) -> float:

        width = abs(width)
        height = abs(height)
        distance = abs(distance)

        if width <= 0:
            return 0.0

        if height <= 0:
            return 0.0

        distance = max(
            distance,
            1e-12,
        )

        numerator = (
            width * height
        )

        denominator = (
            distance
            * math.sqrt(
                distance * distance
                + width * width
                + height * height
            )
        )

        return (
            4.0
            * math.atan(
                numerator
                / denominator
            )
        )

    @staticmethod
    def fraction_of_sphere(
        solid_angle: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                solid_angle
                / (4.0 * math.pi),
            ),
        )


# ============================================================
# Edge weight
# ============================================================

@dataclass
class EdgeWeight:

    distance: float

    distance_squared: float

    solid_angle: float

    opening_fraction: float

    weight: float


class EdgeWeightCalculator:

    def calculate(
        self,
        room_a: Room,
        room_b: Room,
        opening: Opening,
    ) -> EdgeWeight:

        # ----------------------------------------------------
        # Distance between room centers.
        # ----------------------------------------------------

        distance = (
            room_a.center.distance(
                room_b.center
            )
        )

        if distance <= 0:

            distance = (
                Config.DEFAULT_DISTANCE
            )

        distance_squared = (
            distance * distance
        )

        # ----------------------------------------------------
        # Opening dimensions.
        # ----------------------------------------------------

        width = opening.width
        height = opening.height

        if height <= 0:
            height = width

        # ----------------------------------------------------
        # Solid angle.
        # ----------------------------------------------------

        solid_angle = (
            SolidAngleCalculator.rectangular(
                width,
                height,
                distance,
            )
        )

        opening_fraction = (
            SolidAngleCalculator
            .fraction_of_sphere(
                solid_angle
            )
        )

        opening_fraction = max(
            opening_fraction,
            Config.MIN_OPENING_FRACTION,
        )

        # ----------------------------------------------------
        # INVERSE WEIGHT
        #
        #       f
        # w = -------
        #       d^2
        # ----------------------------------------------------

        weight = (
            opening_fraction
            / distance_squared
        )

        return EdgeWeight(
            distance=distance,
            distance_squared=(
                distance_squared
            ),
            solid_angle=solid_angle,
            opening_fraction=(
                opening_fraction
            ),
            weight=weight,
        )


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

    def as_dict(self) -> dict[str, Any]:

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

    def as_dict(self) -> dict[str, Any]:

        return {
            "source": self.source,
            "target": self.target,
            "connection_type": (
                self.connection_type
            ),
            "closable": self.closable,
            "opening_id": self.opening_id,
            "distance": self.distance,
            "distance_squared": (
                self.distance_squared
            ),
            "solid_angle": (
                self.solid_angle
            ),
            "opening_fraction": (
                self.opening_fraction
            ),
            "weight": self.weight,
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

        self.nodes[room.id] = GraphNode(
            id=room.id,
            name=room.name,
            area=room.area,
            x=room.center.x,
            y=room.center.y,
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

    def build(
        self,
        objects: list[IFCObject],
    ) -> RoomGraph:

        rooms = [
            Room(obj)
            for obj in objects
            if obj.is_space()
        ]

        graph = RoomGraph()

        for room in rooms:

            graph.add_room(
                room
            )

        

        # ----------------------------------------------------
        # DEBUG: inspect all IfcOpeningElement dimensions
        # ----------------------------------------------------

        print()
        print(
            "IfcOpeningElement dimensions:"
        )

        for obj in objects:

            if obj.ifc_type != "IfcOpeningElement":
                continue

            opening = Opening(
                object=obj,
                opening_type=OpeningType.OPENING,
            )

            print(
                f"  {obj.id}: "
                f"width={opening.width:.3f}, "
                f"height={opening.height:.3f}, "
                f"name={obj.name!r}"
            )

        print()

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

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        print()
        print(
            "Graph builder:"
        )

        print(
            f"  Rooms: "
            f"{len(rooms)}"
        )

        print(
            f"  Openings: "
            f"{len(openings)}"
        )

        type_counts = {}

        for opening in openings:

            key = (
                opening.opening_type.value
            )

            type_counts[key] = (
                type_counts.get(
                    key,
                    0,
                ) + 1
            )

        for key, count in sorted(
            type_counts.items()
        ):

            print(
                f"    {key}: {count}"
            )

        print()
        print(
            "Opening associations:"
        )

        # ----------------------------------------------------
        # Process openings
        # ----------------------------------------------------

        for opening in openings:

            # Windows do not connect rooms.
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

            print(
                f"  {opening.id}"
                f" [{opening.opening_type.value}]"
            )

            print(
                f"      center="
                f"({opening.center.x:.2f}, "
                f"{opening.center.y:.2f})"
            )

            print(
                f"      size="
                f"{opening.width:.3f}"
                f" x "
                f"{opening.height:.3f}"
            )

            if not association.valid:

                print(
                    "      -> NO ROOM ASSOCIATION"
                )

                continue

            room_a = (
                association.rooms[0]
            )

            room_b = (
                association.rooms[1]
            )

            print(
                f"      -> "
                f"{room_a.id}"
                f" <-> "
                f"{room_b.id}"
            )

            print(
                f"      boundary distances="
                f"{association.distances[0]:.3f}, "
                f"{association.distances[1]:.3f}"
            )

            edge_weight = (
                self.weight_calculator.calculate(
                    room_a,
                    room_b,
                    opening,
                )
            )

            closable = (
                opening.opening_type
                == OpeningType.DOOR
            )

            edge = GraphEdge(

                source=room_a.id,

                target=room_b.id,

                connection_type=(
                    opening.opening_type.value
                ),

                closable=closable,

                opening_id=opening.id,

                distance=(
                    edge_weight.distance
                ),

                distance_squared=(
                    edge_weight.distance_squared
                ),

                solid_angle=(
                    edge_weight.solid_angle
                ),

                opening_fraction=(
                    edge_weight.opening_fraction
                ),

                weight=(
                    edge_weight.weight
                ),
            )

            graph.add_edge(
                edge
            )

            print(
                f"      distance="
                f"{edge_weight.distance:.3f}"
            )

            print(
                f"      solid angle="
                f"{edge_weight.solid_angle:.6g}"
            )

            print(
                f"      fraction="
                f"{edge_weight.opening_fraction:.6g}"
            )

            print(
                f"      weight="
                f"{edge_weight.weight:.6g}"
            )

        return graph


# ============================================================
# Graph analyzer
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
            node_id: []
            for node_id in self.graph.nodes
        }

        for edge in self.graph.edges:

            if (
                doors_closed
                and edge.closable
            ):

                continue

            adjacency[
                edge.source
            ].append(
                edge.target
            )

            adjacency[
                edge.target
            ].append(
                edge.source
            )

        visited = set()

        result = []

        for start in adjacency:

            if start in visited:
                continue

            component = []

            stack = [start]

            visited.add(
                start
            )

            while stack:

                current = stack.pop()

                component.append(
                    current
                )

                for neighbour in adjacency[
                    current
                ]:

                    if neighbour in visited:
                        continue

                    visited.add(
                        neighbour
                    )

                    stack.append(
                        neighbour
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

    """
    Creates the weighted adjacency matrix for doors.

    Example:

                 roomA   roomB   roomC

        roomA     0       0.5     0

        roomB     0.5     0       0.2

        roomC     0       0.2     0

    If multiple doors connect the same two rooms,
    their weights are summed.
    """

    def __init__(
        self,
        graph: RoomGraph,
    ):

        self.graph = graph

    def create(
        self,
    ):

        try:

            import pandas as pd

        except ImportError as exc:

            raise RuntimeError(
                "pandas is required. "
                "Install with: "
                "pip install pandas"
            ) from exc

        room_ids = list(
            self.graph.nodes.keys()
        )

        matrix = pd.DataFrame(
            0.0,
            index=room_ids,
            columns=room_ids,
        )

        for edge in self.graph.edges:

            if not edge.closable:
                continue

            if (
                edge.source not in matrix.index
                or edge.target not in matrix.columns
            ):

                print(
                    "WARNING: edge references "
                    "unknown room:"
                    f" {edge.source} "
                    f"<-> {edge.target}"
                )

                continue

            # Symmetric matrix.
            matrix.loc[
                edge.source,
                edge.target
            ] += edge.weight

            matrix.loc[
                edge.target,
                edge.source
            ] += edge.weight

        return matrix

    def write(
        self,
        filename: str | Path,
    ) -> None:

        try:

            import pyarrow as pa
            import pyarrow.parquet as pq

        except ImportError as exc:

            raise RuntimeError(
                "pyarrow is required. "
                "Install with: "
                "pip install pyarrow"
            ) from exc

        matrix = self.create()

        # ----------------------------------------------------
        # Critical diagnostic.
        # ----------------------------------------------------

        nonzero = (
            matrix.to_numpy() != 0
        ).sum()

        print()
        print(
            "Closable matrix:"
        )

        print(
            f"  Dimensions: "
            f"{matrix.shape[0]} x "
            f"{matrix.shape[1]}"
        )

        print(
            f"  Non-zero cells: "
            f"{nonzero}"
        )

        if nonzero == 0:

            print(
                "  WARNING:"
            )

            print(
                "  The matrix contains no "
                "closable connections."
            )

            print(
                "  Number of closable graph edges: "
                f"{len(self.graph.closable_edges())}"
            )

        else:

            print()
            print(matrix)

        # ----------------------------------------------------
        # Convert DataFrame to Arrow.
        # ----------------------------------------------------

        output = matrix.reset_index()

        output = output.rename(
            columns={
                output.columns[0]:
                    "room_id"
            }
        )

        table = pa.Table.from_pandas(
            output,
            preserve_index=False,
        )

        pq.write_table(
            table,
            filename,
        )

        print()
        print(
            f"Wrote: {filename}"
        )


# ============================================================
# JSON writer
# ============================================================

class RoomGraphJsonWriter:

    def write(
        self,
        graph: RoomGraph,
        analyzer: GraphAnalyzer,
        filename: str | Path,
    ) -> None:

        data = graph.as_dict()

        data["analysis"] = {

            "components_all":
                analyzer.components(
                    doors_closed=False
                ),

            "components_doors_closed":
                analyzer.components(
                    doors_closed=True
                ),
        }

        data["weight_definition"] = {

            "formula":
                "opening_fraction / distance^2",

            "opening_fraction":
                "solid_angle / (4*pi)",

            "solid_angle":
                (
                    "4*atan(w*h / "
                    "(d*sqrt(d^2+w^2+h^2)))"
                ),
        }

        data["matrix"] = {

            "filename":
                Config.OUTPUT_MATRIX,

            "type":
                "closable",

            "meaning":
                "weighted adjacency matrix of doors",

            "symmetric":
                True,
        }

        with Path(filename).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Wrote graph JSON: "
            f"{filename}"
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

        try:

            import matplotlib.pyplot as plt

        except ImportError as exc:

            raise RuntimeError(
                "matplotlib is required. "
                "Install with: "
                "pip install matplotlib"
            ) from exc

        if not graph.nodes:

            print(
                "No graph nodes."
            )

            return

        fig, ax = plt.subplots(
            figsize=(12, 9)
        )

        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        for edge in graph.edges:

            if edge.source not in graph.nodes:
                continue

            if edge.target not in graph.nodes:
                continue

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

            ax.plot(
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

            mx = (
                source.x
                + target.x
            ) / 2.0

            my = (
                source.y
                + target.y
            ) / 2.0

            ax.text(
                mx,
                my,
                (
                    f"{edge.connection_type}\n"
                    f"w={edge.weight:.3g}"
                ),
                fontsize=8,
                ha="center",
                va="center",
            )

        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

        for node in graph.nodes.values():

            ax.scatter(
                node.x,
                node.y,
                s=Config.NODE_SIZE,
                zorder=3,
            )

            ax.text(
                node.x,
                node.y,
                (
                    f"{node.name}\n"
                    f"{node.id}"
                ),
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )

        ax.set_title(
            "Room connectivity graph"
        )

        ax.set_xlabel(
            "X"
        )

        ax.set_ylabel(
            "Y"
        )

        ax.set_aspect(
            "equal",
            adjustable="datalim",
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            filename,
            dpi=Config.IMAGE_DPI,
        )

        plt.close(
            fig
        )

        print(
            f"Wrote graph image: "
            f"{filename}"
        )


# ============================================================
# Application
# ============================================================

class RoomGraphApplication:

    def __init__(
        self,
        input_filename: str | Path,
        output_json: str | Path,
        output_image: str | Path,
        output_matrix: str | Path,
    ):

        self.input_filename = Path(
            input_filename
        )

        self.output_json = Path(
            output_json
        )

        self.output_image = Path(
            output_image
        )

        self.output_matrix = Path(
            output_matrix
        )

    def run(self) -> None:

        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        reader = IFCJsonReader(
            self.input_filename
        )

        objects = reader.read()

        print(
            f"Total IFC objects: "
            f"{len(objects)}"
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        type_counts = {}

        for obj in objects:

            type_counts[
                obj.ifc_type
            ] = (
                type_counts.get(
                    obj.ifc_type,
                    0,
                ) + 1
            )

        print()
        print(
            "IFC types:"
        )

        for ifc_type, count in sorted(
            type_counts.items()
        ):

            print(
                f"  {ifc_type}: {count}"
            )

        spaces = [
            obj
            for obj in objects
            if obj.is_space()
        ]

        doors = [
            obj
            for obj in objects
            if obj.is_door()
        ]

        windows = [
            obj
            for obj in objects
            if obj.is_window()
        ]

        openings = [
            obj
            for obj in objects
            if obj.is_opening()
        ]

        print()
        print(
            "Semantic objects:"
        )

        print(
            f"  Spaces:   {len(spaces)}"
        )

        print(
            f"  Doors:    {len(doors)}"
        )

        print(
            f"  Windows:  {len(windows)}"
        )

        print(
            f"  Openings: {len(openings)}"
        )

        # ----------------------------------------------------
        # IMPORTANT diagnostic
        # ----------------------------------------------------

        if len(doors) == 0:

            print()
            print(
                "WARNING: ZERO IfcDoor objects found."
            )

            print(
                "Therefore matrixclose.parquet "
                "will contain only zeros."
            )

            print()
            print(
                "Detected IFC types were:"
            )

            for ifc_type in sorted(
                type_counts
            ):

                print(
                    f"  {ifc_type}"
                )

        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        builder = RoomGraphBuilder()

        graph = builder.build(
            objects
        )

        # ----------------------------------------------------
        # Graph statistics
        # ----------------------------------------------------

        print()
        print(
            "Final graph:"
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
            f"  Permanent/open edges: "
            f"{len(graph.open_edges())}"
        )

        # ----------------------------------------------------
        # Write graph JSON
        # ----------------------------------------------------

        analyzer = GraphAnalyzer(
            graph
        )

        json_writer = (
            RoomGraphJsonWriter()
        )

        json_writer.write(
            graph=graph,
            analyzer=analyzer,
            filename=self.output_json,
        )

        # ----------------------------------------------------
        # Write closable matrix
        # ----------------------------------------------------

        matrix = (
            ClosableGraphMatrix(
                graph
            )
        )

        matrix.write(
            self.output_matrix
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        visualizer = (
            GraphVisualizer()
        )

        visualizer.visualize(
            graph=graph,
            filename=self.output_image,
        )

        # ----------------------------------------------------
        # Final summary
        # ----------------------------------------------------

        print()
        print(
            "Finished."
        )

        print(
            f"  JSON:   {self.output_json}"
        )

        print(
            f"  Image:  {self.output_image}"
        )

        print(
            f"  Matrix: {self.output_matrix}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build a weighted room graph "
            "from IFC JSON."
        )
    )

    parser.add_argument(
        "input",
        help="Input IFC JSON file",
    )

    parser.add_argument(
        "--output",
        default=Config.OUTPUT_JSON,
        help=(
            "Output graph JSON "
            "(default: room_graph.json)"
        ),
    )

    parser.add_argument(
        "--visualization",
        default=Config.OUTPUT_IMAGE,
        help=(
            "Output graph visualization "
            "(default: room_graph.png)"
        ),
    )

    parser.add_argument(
        "--matrix",
        default=Config.OUTPUT_MATRIX,
        help=(
            "Output closable matrix "
            "(default: matrixclose.parquet)"
        ),
    )

    args = parser.parse_args()

    application = RoomGraphApplication(
        input_filename=args.input,
        output_json=args.output,
        output_image=args.visualization,
        output_matrix=args.matrix,
    )

    application.run()


if __name__ == "__main__":
    main()
