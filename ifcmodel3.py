#!/usr/bin/env python3

"""
IFC-semantic architectural floor-plan analyzer.

Pipeline
--------

1. Load drawing
2. Detect edges
3. Detect raw line segments
4. Reconstruct interrupted / collinear lines
5. Detect whitespace-separated regions
6. Analyze geometry independently per region
7. Detect walls, spaces, doors, windows, openings, stairs
8. OCR text
9. Detect dimension systems
10. Associate dimension endpoints with geometry
11. Estimate drawing scale
12. Calculate width / height / length / area
13. Assign IFC semantics
14. Export semantic catalog as JSON
15. Create visualization

Dependencies

    pip install opencv-python numpy pytesseract

macOS:

    brew install tesseract

Example:

    python ifcmodel1.py page_1.png

    python ifcmodel1.py page_1.png \
        --output catalog.json \
        --visualization result.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import uuid

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """
    Central configuration.

    Values are primarily in pixels unless explicitly stated.
    """

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    CANNY_LOW = 50
    CANNY_HIGH = 150

    # --------------------------------------------------------
    # Hough
    # --------------------------------------------------------

    HOUGH_THRESHOLD = 35
    HOUGH_MIN_LINE_LENGTH = 15
    HOUGH_MAX_LINE_GAP = 8

    # --------------------------------------------------------
    # Line reconstruction
    # --------------------------------------------------------

    ANGLE_TOLERANCE_DEG = 3.0
    COLLINEAR_DISTANCE = 8.0
    MERGE_GAP = 25.0

    # --------------------------------------------------------
    # Region detection
    # --------------------------------------------------------

    REGION_CLOSE_KERNEL = 5
    REGION_LINE_THICKNESS = 3
    REGION_MIN_AREA = 800

    # --------------------------------------------------------
    # Walls
    # --------------------------------------------------------

    MIN_WALL_THICKNESS = 30
    MAX_WALL_THICKNESS = 120
    WALL_ANGLE_TOLERANCE = 4.0
    WALL_OVERLAP_RATIO = 0.45

    # --------------------------------------------------------
    # Doors / openings
    # --------------------------------------------------------

    MAX_OPENING_GAP = 150
    MIN_OPENING_GAP = 20

    # --------------------------------------------------------
    # Stairs
    # --------------------------------------------------------

    STAIR_MIN_LINES = 4
    STAIR_SPACING_TOLERANCE = 0.30

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    OCR_SCALE = 2.0
    OCR_PSM = 11

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    DIMENSION_MIN_LENGTH = 25
    DIMENSION_MAX_TEXT_DISTANCE = 100

    DIMENSION_ENDPOINT_DISTANCE = 35
    DIMENSION_OBJECT_DISTANCE = 50

    DIMENSION_ANGLE_TOLERANCE = 8.0

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    DEBUG = True


# ============================================================
# IFC SEMANTICS
# ============================================================

class IFCType(str, Enum):

    IFC_WALL = "IfcWall"
    IFC_WALL_STANDARD_CASE = "IfcWallStandardCase"

    IFC_DOOR = "IfcDoor"
    IFC_WINDOW = "IfcWindow"

    IFC_SPACE = "IfcSpace"
    IFC_STAIR = "IfcStair"

    IFC_OPENING_ELEMENT = "IfcOpeningElement"

    IFC_SLAB = "IfcSlab"
    IFC_ROOF = "IfcRoof"
    IFC_COLUMN = "IfcColumn"
    IFC_BEAM = "IfcBeam"

    IFC_FURNISHING_ELEMENT = "IfcFurnishingElement"

    IFC_BUILDING_ELEMENT_PROXY = "IfcBuildingElementProxy"

    IFC_ANNOTATION = "IfcAnnotation"
    IFC_DIMENSION = "IfcDimension"

    IFC_UNKNOWN = "IfcBuildingElementProxy"


# ============================================================
# BASIC GEOMETRY
# ============================================================

@dataclass
class Point:

    x: float
    y: float

    def distance(
        self,
        other: "Point"
    ) -> float:

        return math.hypot(
            self.x - other.x,
            self.y - other.y
        )

    def to_tuple(self):

        return (
            self.x,
            self.y
        )


@dataclass
class BoundingBox:

    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self):

        return self.x + self.width

    @property
    def y2(self):

        return self.y + self.height

    @property
    def area(self):

        return (
            self.width
            * self.height
        )

    @property
    def center(self):

        return Point(
            self.x + self.width / 2,
            self.y + self.height / 2
        )

    def contains(
        self,
        point: Point
    ) -> bool:

        return (
            self.x <= point.x <= self.x2
            and
            self.y <= point.y <= self.y2
        )


@dataclass
class Segment2D:

    p1: Point
    p2: Point

    @property
    def length(self):

        return self.p1.distance(
            self.p2
        )

    @property
    def angle_deg(self):

        angle = math.degrees(
            math.atan2(
                self.p2.y - self.p1.y,
                self.p2.x - self.p1.x
            )
        )

        while angle < 0:
            angle += 180

        while angle >= 180:
            angle -= 180

        return angle

    @property
    def midpoint(self):

        return Point(
            (self.p1.x + self.p2.x) / 2,
            (self.p1.y + self.p2.y) / 2
        )


@dataclass
class Gap:

    line: Segment2D

    start: float
    end: float

    @property
    def length(self):

        return self.end - self.start


# ============================================================
# GENERAL DISTANCE HELPER
# ============================================================

def distance(
    p1: Point,
    p2: Point
) -> float:
    """
    Euclidean distance between two points.

    Kept as a standalone helper because several geometry
    algorithms use it directly.
    """

    return math.hypot(
        float(p2.x) - float(p1.x),
        float(p2.y) - float(p1.y)
    )


# ============================================================
# GEOMETRY MATH
# ============================================================

class GeometryMath:

    @staticmethod
    def angle_difference(
        a: float,
        b: float
    ):

        d = abs(a - b)

        return min(
            d,
            180 - d
        )

    @staticmethod
    def bounding_box(
        points: list[Point]
    ) -> BoundingBox:

        if not points:

            return BoundingBox(
                0,
                0,
                0,
                0
            )

        xs = [p.x for p in points]
        ys = [p.y for p in points]

        return BoundingBox(
            min(xs),
            min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys)
        )

    @staticmethod
    def polygon_area(
        points: list[Point]
    ) -> float:

        if len(points) < 3:
            return 0.0

        result = 0.0

        for i, p in enumerate(points):

            q = points[
                (i + 1) % len(points)
            ]

            result += (
                p.x * q.y
                - q.x * p.y
            )

        return abs(result) / 2.0

    @staticmethod
    def point_line_distance(
        point: Point,
        line: Segment2D
    ) -> float:

        dx = line.p2.x - line.p1.x
        dy = line.p2.y - line.p1.y

        denominator = math.hypot(
            dx,
            dy
        )

        if denominator < 1e-9:

            return point.distance(
                line.p1
            )

        return abs(
            dy * point.x
            - dx * point.y
            + line.p2.x * line.p1.y
            - line.p2.y * line.p1.x
        ) / denominator

    @staticmethod
    def point_segment_distance(
        point: Point,
        line: Segment2D
    ) -> float:

        dx = line.p2.x - line.p1.x
        dy = line.p2.y - line.p1.y

        denominator = (
            dx * dx
            + dy * dy
        )

        if denominator < 1e-9:

            return point.distance(
                line.p1
            )

        t = (
            (point.x - line.p1.x) * dx
            +
            (point.y - line.p1.y) * dy
        ) / denominator

        t = max(
            0.0,
            min(1.0, t)
        )

        projection = Point(
            line.p1.x + t * dx,
            line.p1.y + t * dy
        )

        return point.distance(
            projection
        )

    @staticmethod
    def project_point(
        point: Point,
        line: Segment2D
    ) -> tuple[Point, float]:

        dx = line.p2.x - line.p1.x
        dy = line.p2.y - line.p1.y

        length2 = (
            dx * dx
            + dy * dy
        )

        if length2 < 1e-9:

            return line.p1, 0.0

        t = (
            (point.x - line.p1.x) * dx
            +
            (point.y - line.p1.y) * dy
        ) / length2

        projection = Point(
            line.p1.x + t * dx,
            line.p1.y + t * dy
        )

        return projection, t

    @staticmethod
    def line_intersection(
        a: Segment2D,
        b: Segment2D
    ) -> Optional[Point]:

        x1, y1 = a.p1.x, a.p1.y
        x2, y2 = a.p2.x, a.p2.y

        x3, y3 = b.p1.x, b.p1.y
        x4, y4 = b.p2.x, b.p2.y

        denominator = (
            (x1 - x2) * (y3 - y4)
            -
            (y1 - y2) * (x3 - x4)
        )

        if abs(denominator) < 1e-9:

            return None

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            -
            (x1 - x2)
            * (x3 * y4 - y3 * x4)
        ) / denominator

        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            -
            (y1 - y2)
            * (x3 * y4 - y3 * x4)
        ) / denominator

        if (
            min(x1, x2) - 1 <= px <= max(x1, x2) + 1
            and
            min(y1, y2) - 1 <= py <= max(y1, y2) + 1
            and
            min(x3, x4) - 1 <= px <= max(x3, x4) + 1
            and
            min(y3, y4) - 1 <= py <= max(y3, y4) + 1
        ):

            return Point(
                px,
                py
            )

        return None

    @staticmethod
    def overlap_ratio(
        a: Segment2D,
        b: Segment2D
    ) -> float:

        if GeometryMath.angle_difference(
            a.angle_deg,
            b.angle_deg
        ) > 5:

            return 0.0

        angle = math.radians(
            a.angle_deg
        )

        ux = math.cos(angle)
        uy = math.sin(angle)

        origin = a.p1

        def project(p):

            return (
                (p.x - origin.x) * ux
                +
                (p.y - origin.y) * uy
            )

        a1 = project(a.p1)
        a2 = project(a.p2)

        b1 = project(b.p1)
        b2 = project(b.p2)

        amin = min(a1, a2)
        amax = max(a1, a2)

        bmin = min(b1, b2)
        bmax = max(b1, b2)

        overlap = max(
            0,
            min(amax, bmax)
            -
            max(amin, bmin)
        )

        denominator = min(
            amax - amin,
            bmax - bmin
        )

        if denominator <= 0:

            return 0.0

        return overlap / denominator


# ============================================================
# IFC PROPERTY SET
# ============================================================

@dataclass
class IFCProperties:

    name: Optional[str] = None

    description: Optional[str] = None

    object_type: Optional[str] = None

    width: Optional[float] = None
    height: Optional[float] = None
    length: Optional[float] = None
    area: Optional[float] = None

    properties: dict = field(
        default_factory=dict
    )


# ============================================================
# IFC OBJECT
# ============================================================

@dataclass
class IFCObject:

    id: str

    ifc_type: IFCType

    name: str

    geometry: list[Point] = field(
        default_factory=list
    )

    bbox: Optional[BoundingBox] = None

    angle_deg: float = 0.0

    width_px: float = 0.0
    height_px: float = 0.0
    length_px: float = 0.0
    area_px2: float = 0.0

    width_m: Optional[float] = None
    height_m: Optional[float] = None
    length_m: Optional[float] = None
    area_m2: Optional[float] = None

    properties: IFCProperties = field(
        default_factory=IFCProperties
    )

    references: list[str] = field(
        default_factory=list
    )

    referenced_by: list[str] = field(
        default_factory=list
    )

    segment_id: Optional[str] = None

    confidence: float = 0.0


# ============================================================
# TEXT
# ============================================================

@dataclass
class TextObject:

    id: str

    text: str

    bbox: BoundingBox

    confidence: float = 0.0

    references: list[str] = field(
        default_factory=list
    )


# ============================================================
# DIMENSION
# ============================================================

@dataclass
class DimensionObject:

    id: str

    value: float

    unit: str

    value_m: float

    bbox: BoundingBox

    dimension_line: Segment2D

    orientation: str

    dimension_type: str

    extension_lines: list[Segment2D] = field(
        default_factory=list
    )

    endpoint1: Optional[Point] = None
    endpoint2: Optional[Point] = None

    reference1: Optional[str] = None
    reference2: Optional[str] = None

    confidence: float = 0.0


# ============================================================
# PLAN REGION
# ============================================================

@dataclass
class PlanRegion:

    id: str

    bbox: BoundingBox

    area_px2: float

    geometry_ids: list[str] = field(
        default_factory=list
    )


# ============================================================
# OBJECT CATALOG
# ============================================================

class ObjectCatalog:

    def __init__(self):

        self.objects: dict[str, IFCObject] = {}
        self.texts: dict[str, TextObject] = {}
        self.dimensions: dict[str, DimensionObject] = {}
        self.regions: dict[str, PlanRegion] = {}

    @staticmethod
    def new_id(
        prefix: str
    ):

        return (
            f"{prefix}_"
            f"{uuid.uuid4().hex[:8]}"
        )

    def add_object(
        self,
        obj: IFCObject
    ):

        self.objects[obj.id] = obj

    def add_text(
        self,
        text: TextObject
    ):

        self.texts[text.id] = text

    def add_dimension(
        self,
        dimension: DimensionObject
    ):

        self.dimensions[
            dimension.id
        ] = dimension

    def add_region(
        self,
        region: PlanRegion
    ):

        self.regions[
            region.id
        ] = region

    def add_reference(
        self,
        source_id: str,
        target_id: str
    ):

        if source_id in self.objects:

            source = self.objects[
                source_id
            ]

        elif source_id in self.texts:

            source = self.texts[
                source_id
            ]

        elif source_id in self.dimensions:

            source = self.dimensions[
                source_id
            ]

        else:

            return

        if target_id not in source.references:

            source.references.append(
                target_id
            )

        if target_id in self.objects:

            target = self.objects[
                target_id
            ]

            if source_id not in target.referenced_by:

                target.referenced_by.append(
                    source_id
                )


# ============================================================
# IMAGE PROCESSOR
# ============================================================

class ImageProcessor:

    def __init__(
        self,
        config: Config
    ):

        self.config = config

    def load(
        self,
        filename: str
    ):

        image = cv2.imread(
            filename,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise FileNotFoundError(
                f"Cannot read image: {filename}"
            )

        return image

    def edges(
        self,
        image
    ):

        return cv2.Canny(
            image,
            self.config.CANNY_LOW,
            self.config.CANNY_HIGH
        )


# ============================================================
# LINE DETECTOR
# ============================================================

class LineDetector:

    def __init__(
        self,
        config: Config
    ):

        self.config = config

    def detect(
        self,
        edges
    ) -> list[Segment2D]:

        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.config.HOUGH_THRESHOLD,
            minLineLength=self.config.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=self.config.HOUGH_MAX_LINE_GAP
        )

        result: list[Segment2D] = []

        if raw_lines is None:

            return result

        # OpenCV commonly returns:
        #
        #   (N, 1, 4)
        #
        # but depending on version / processing it can
        # also be represented differently.
        #
        # Normalize everything to:
        #
        #   (N, 4)
        #
        raw_lines = np.asarray(
            raw_lines
        ).reshape(-1, 4)

        for values in raw_lines:

            x1, y1, x2, y2 = map(
                float,
                values
            )

            p1 = Point(
                x1,
                y1
            )

            p2 = Point(
                x2,
                y2
            )

            length = distance(
                p1,
                p2
            )

            if (
                length
                <
                self.config.HOUGH_MIN_LINE_LENGTH
            ):

                continue

            result.append(
                Segment2D(
                    p1,
                    p2
                )
            )

        return result


# ============================================================
# LINE RECONSTRUCTOR
# ============================================================

class LineReconstructor:

    def __init__(
        self,
        config: Config
    ):

        self.config = config

    def reconstruct(
        self,
        lines: list[Segment2D]
    ) -> list[Segment2D]:

        groups: list[
            list[Segment2D]
        ] = []

        for line in lines:

            found = False

            for group in groups:

                if self.compatible(
                    line,
                    group[0]
                ):

                    group.append(line)

                    found = True

                    break

            if not found:

                groups.append(
                    [line]
                )

        result = []

        for group in groups:

            result.extend(
                self.merge_group(
                    group
                )
            )

        return result

    def compatible(
        self,
        a: Segment2D,
        b: Segment2D
    ) -> bool:

        if GeometryMath.angle_difference(
            a.angle_deg,
            b.angle_deg
        ) > self.config.ANGLE_TOLERANCE_DEG:

            return False

        distance_from_line = (
            GeometryMath.point_line_distance(
                b.p1,
                a
            )
        )

        return (
            distance_from_line
            <= self.config.COLLINEAR_DISTANCE
        )

    def merge_group(
        self,
        group: list[Segment2D]
    ) -> list[Segment2D]:

        if not group:

            return []

        base = group[0]

        angle = math.radians(
            base.angle_deg
        )

        ux = math.cos(angle)
        uy = math.sin(angle)

        ox = base.p1.x
        oy = base.p1.y

        intervals = []

        for line in group:

            t1 = (
                (line.p1.x - ox) * ux
                +
                (line.p1.y - oy) * uy
            )

            t2 = (
                (line.p2.x - ox) * ux
                +
                (line.p2.y - oy) * uy
            )

            intervals.append(
                (
                    min(t1, t2),
                    max(t1, t2)
                )
            )

        intervals.sort()

        result = []

        start, end = intervals[0]

        for a, b in intervals[1:]:

            if (
                a
                <=
                end + self.config.MERGE_GAP
            ):

                end = max(
                    end,
                    b
                )

            else:

                result.append(
                    self.make_segment(
                        ox,
                        oy,
                        ux,
                        uy,
                        start,
                        end
                    )
                )

                start = a
                end = b

        result.append(
            self.make_segment(
                ox,
                oy,
                ux,
                uy,
                start,
                end
            )
        )

        return result

    @staticmethod
    def make_segment(
        ox,
        oy,
        ux,
        uy,
        start,
        end
    ) -> Segment2D:

        return Segment2D(
            Point(
                ox + ux * start,
                oy + uy * start
            ),
            Point(
                ox + ux * end,
                oy + uy * end
            )
        )


# ============================================================
# GAP DETECTOR
# ============================================================

class GapDetector:

    """
    Finds gaps along reconstructed lines.

    Important:
    A gap is measured along one collinear line.
    The distance between parallel walls is NOT a gap.
    """

    def __init__(
        self,
        config: Config
    ):

        self.config = config

    def detect(
        self,
        original: list[Segment2D],
        reconstructed: list[Segment2D]
    ) -> list[Gap]:

        result = []

        for master in reconstructed:

            compatible = []

            for line in original:

                if GeometryMath.angle_difference(
                    master.angle_deg,
                    line.angle_deg
                ) > self.config.ANGLE_TOLERANCE_DEG:

                    continue

                line_distance = (
                    GeometryMath.point_line_distance(
                        line.p1,
                        master
                    )
                )

                if (
                    line_distance
                    <= self.config.COLLINEAR_DISTANCE
                ):

                    compatible.append(
                        line
                    )

            if not compatible:

                continue

            angle = math.radians(
                master.angle_deg
            )

            ux = math.cos(angle)
            uy = math.sin(angle)

            origin = master.p1

            def project(p):

                return (
                    (p.x - origin.x) * ux
                    +
                    (p.y - origin.y) * uy
                )

            intervals = []

            for line in compatible:

                a = project(
                    line.p1
                )

                b = project(
                    line.p2
                )

                intervals.append(
                    (
                        min(a, b),
                        max(a, b)
                    )
                )

            if not intervals:

                continue

            intervals.sort()

            cursor = intervals[0][0]

            for a, b in intervals[1:]:

                if a > cursor:

                    gap_length = (
                        a - cursor
                    )

                    if (
                        gap_length
                        >=
                        self.config.MIN_OPENING_GAP
                    ):

                        result.append(
                            Gap(
                                master,
                                cursor,
                                a
                            )
                        )

                cursor = max(
                    cursor,
                    b
                )

        return result


# ============================================================
# PLAN REGION DETECTOR
# ============================================================

class PlanRegionDetector:

    """
    Detect whitespace-separated regions.

    Lines are rasterized first. The inverse image is then
    searched for connected white regions.
    """

    def __init__(
        self,
        config: Config
    ):

        self.config = config

    def detect(
        self,
        image_shape,
        lines: list[Segment2D],
        catalog: ObjectCatalog
    ):

        height, width = image_shape[:2]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        thickness = (
            self.config.REGION_LINE_THICKNESS
        )

        for line in lines:

            cv2.line(
                mask,
                (
                    int(round(line.p1.x)),
                    int(round(line.p1.y))
                ),
                (
                    int(round(line.p2.x)),
                    int(round(line.p2.y))
                ),
                255,
                thickness
            )

        kernel_size = (
            self.config.REGION_CLOSE_KERNEL
        )

        kernel = np.ones(
            (
                kernel_size,
                kernel_size
            ),
            dtype=np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        inverse = cv2.bitwise_not(
            mask
        )

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                inverse,
                connectivity=8
            )
        )

        regions = []

        for i in range(
            1,
            count
        ):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if (
                area
                <
                self.config.REGION_MIN_AREA
            ):

                continue

            x = stats[
                i,
                cv2.CC_STAT_LEFT
            ]

            y = stats[
                i,
                cv2.CC_STAT_TOP
            ]

            w = stats[
                i,
                cv2.CC_STAT_WIDTH
            ]

            h = stats[
                i,
                cv2.CC_STAT_HEIGHT
            ]

            # Ignore a region touching the complete image
            # border. This is normally the page background.
            touches_border = (
                x <= 0
                or y <= 0
                or x + w >= width - 1
                or y + h >= height - 1
            )

            if touches_border:

                continue

            region = PlanRegion(

                id=catalog.new_id(
                    "space_region"
                ),

                bbox=BoundingBox(
                    float(x),
                    float(y),
                    float(w),
                    float(h)
                ),

                area_px2=float(area)
            )

            catalog.add_region(
                region
            )

            regions.append(
                region
            )

        return regions


# ============================================================
# IFC GEOMETRY FACTORY
# ============================================================

class IFCObjectFactory:

    def __init__(
        self,
        catalog: ObjectCatalog
    ):

        self.catalog = catalog

    def create(
        self,
        ifc_type: IFCType,
        points: list[Point],
        name: str = ""
    ) -> IFCObject:

        bbox = GeometryMath.bounding_box(
            points
        )

        obj = IFCObject(

            id=self.catalog.new_id(
                "ifc"
            ),

            ifc_type=ifc_type,

            name=name,

            geometry=points,

            bbox=bbox
        )

        self.catalog.add_object(
            obj
        )

        return obj


# ============================================================
# GEOMETRY ANALYZER
# ============================================================

class GeometryAnalyzer:

    def __init__(
        self,
        config: Config,
        catalog: ObjectCatalog
    ):

        self.config = config

        self.catalog = catalog

        self.factory = IFCObjectFactory(
            catalog
        )

    def analyze_lines(
        self,
        lines: list[Segment2D]
    ):

        for line in lines:

            obj = self.factory.create(
                IFCType.IFC_BUILDING_ELEMENT_PROXY,
                [
                    line.p1,
                    line.p2
                ],
                "Linear geometry"
            )

            obj.angle_deg = (
                line.angle_deg
            )

            obj.length_px = (
                line.length
            )

            obj.width_px = 1.0
            obj.height_px = 1.0

            obj.properties = IFCProperties(
                object_type="LINE"
            )

    def detect_walls(
        self,
        lines: list[Segment2D]
    ):

        for i, a in enumerate(lines):

            for b in lines[i + 1:]:

                if GeometryMath.angle_difference(
                    a.angle_deg,
                    b.angle_deg
                ) > self.config.WALL_ANGLE_TOLERANCE:

                    continue

                wall_thickness = (
                    GeometryMath.point_line_distance(
                        b.p1,
                        a
                    )
                )

                if not (
                    self.config.MIN_WALL_THICKNESS
                    <= wall_thickness
                    <= self.config.MAX_WALL_THICKNESS
                ):

                    continue

                overlap = GeometryMath.overlap_ratio(
                    a,
                    b
                )

                if (
                    overlap
                    <
                    self.config.WALL_OVERLAP_RATIO
                ):

                    continue

                points = [
                    a.p1,
                    a.p2,
                    b.p2,
                    b.p1
                ]

                obj = self.factory.create(
                    IFCType.IFC_WALL,
                    points,
                    "Wall"
                )

                obj.width_px = (
                    wall_thickness
                )

                obj.length_px = min(
                    a.length,
                    b.length
                )

                obj.area_px2 = (
                    wall_thickness
                    * obj.length_px
                )

                obj.angle_deg = (
                    a.angle_deg
                )

                obj.confidence = (
                    0.5
                    +
                    0.5 * overlap
                )

                obj.properties = IFCProperties(
                    object_type="WALL",
                    properties={
                        "wall_thickness_px":
                            wall_thickness,
                        "parallel_overlap":
                            overlap
                    }
                )

    def detect_spaces(
        self,
        regions: list[PlanRegion]
    ):

        for region in regions:

            obj = self.factory.create(
                IFCType.IFC_SPACE,
                [
                    Point(
                        region.bbox.x,
                        region.bbox.y
                    ),
                    Point(
                        region.bbox.x2,
                        region.bbox.y
                    ),
                    Point(
                        region.bbox.x2,
                        region.bbox.y2
                    ),
                    Point(
                        region.bbox.x,
                        region.bbox.y2
                    )
                ],
                "Space"
            )

            obj.segment_id = (
                region.id
            )

            obj.width_px = (
                region.bbox.width
            )

            obj.height_px = (
                region.bbox.height
            )

            obj.length_px = max(
                region.bbox.width,
                region.bbox.height
            )

            obj.area_px2 = (
                region.area_px2
            )

            obj.properties = IFCProperties(
                object_type="ROOM / SPACE"
            )

            region.geometry_ids.append(
                obj.id
            )

    def detect_stairs(
        self,
        lines: list[Segment2D]
    ):

        horizontal = [
            x
            for x in lines
            if (
                GeometryMath.angle_difference(
                    x.angle_deg,
                    0
                )
                < 5
            )
        ]

        if (
            len(horizontal)
            <
            self.config.STAIR_MIN_LINES
        ):

            return

        groups = []

        for line in horizontal:

            placed = False

            for group in groups:

                if abs(
                    line.midpoint.y
                    -
                    group[0].midpoint.y
                ) < 150:

                    group.append(
                        line
                    )

                    placed = True

                    break

            if not placed:

                groups.append(
                    [line]
                )

        for group in groups:

            if (
                len(group)
                <
                self.config.STAIR_MIN_LINES
            ):

                continue

            ys = sorted(
                x.midpoint.y
                for x in group
            )

            spacings = np.diff(
                ys
            )

            if len(spacings) == 0:

                continue

            median = float(
                np.median(
                    spacings
                )
            )

            if median <= 0:

                continue

            deviation = (
                np.max(
                    np.abs(
                        spacings
                        -
                        median
                    )
                )
                /
                median
            )

            if (
                deviation
                >
                self.config.STAIR_SPACING_TOLERANCE
            ):

                continue

            points = []

            for line in group:

                points.extend([
                    line.p1,
                    line.p2
                ])

            obj = self.factory.create(
                IFCType.IFC_STAIR,
                points,
                "Stair"
            )

            obj.width_px = max(
                x.length
                for x in group
            )

            obj.height_px = (
                ys[-1] - ys[0]
            )

            obj.properties = IFCProperties(
                object_type="STAIR",
                properties={
                    "step_count":
                        len(group),
                    "step_spacing_px":
                        median
                }
            )

            obj.confidence = max(
                0.0,
                min(
                    1.0,
                    1.0 - deviation
                )
            )


# ============================================================
# TEXT CLASSIFIER
# ============================================================

class IFCTextClassifier:

    DOOR_WORDS = {
        "tür",
        "tuer",
        "door"
    }

    WINDOW_WORDS = {
        "fenster",
        "window"
    }

    STAIR_WORDS = {
        "treppe",
        "stairs",
        "stiege"
    }

    SPACE_WORDS = {
        "zimmer",
        "raum",
        "wohnzimmer",
        "küche",
        "kueche",
        "bad",
        "wc",
        "schlafzimmer",
        "büro",
        "buero",
        "flur",
        "gang",
        "korridor"
    }

    @classmethod
    def classify(
        cls,
        text: str
    ) -> Optional[IFCType]:

        normalized = (
            text.lower()
            .strip()
        )

        for word in cls.DOOR_WORDS:

            if word in normalized:

                return IFCType.IFC_DOOR

        for word in cls.WINDOW_WORDS:

            if word in normalized:

                return IFCType.IFC_WINDOW

        for word in cls.STAIR_WORDS:

            if word in normalized:

                return IFCType.IFC_STAIR

        for word in cls.SPACE_WORDS:

            if word in normalized:

                return IFCType.IFC_SPACE

        return None


# ============================================================
# OCR READER
# ============================================================

class OCRReader:

    def __init__(
        self,
        config: Config
    ):

        self.config = config

        try:

            import pytesseract

            self.pytesseract = (
                pytesseract
            )

            self.available = True

        except ImportError:

            self.pytesseract = None

            self.available = False

    def read(
        self,
        image,
        catalog: ObjectCatalog
    ):

        if not self.available:

            print(
                "WARNING: pytesseract not installed"
            )

            return []

        scale = (
            self.config.OCR_SCALE
        )

        enlarged = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        data = self.pytesseract.image_to_data(
            enlarged,
            output_type=(
                self.pytesseract.Output.DICT
            ),
            config=(
                f"--psm "
                f"{self.config.OCR_PSM}"
            )
        )

        result = []

        for i, text in enumerate(
            data["text"]
        ):

            text = text.strip()

            if not text:

                continue

            try:

                confidence = float(
                    data["conf"][i]
                )

            except (
                ValueError,
                TypeError
            ):

                confidence = 0.0

            bbox = BoundingBox(

                float(
                    data["left"][i]
                ) / scale,

                float(
                    data["top"][i]
                ) / scale,

                float(
                    data["width"][i]
                ) / scale,

                float(
                    data["height"][i]
                ) / scale
            )

            obj = TextObject(

                id=catalog.new_id(
                    "text"
                ),

                text=text,

                bbox=bbox,

                confidence=confidence
            )

            catalog.add_text(
                obj
            )

            result.append(
                obj
            )

        return result


# ============================================================
# DIMENSION PARSER
# ============================================================

class DimensionParser:

    PATTERN = re.compile(
        r"""
        (?P<value>
            \d+(?:[.,]\d+)?
        )
        \s*
        (?P<unit>
            mm|cm|m
        )?
        """,
        re.IGNORECASE | re.VERBOSE
    )

    FACTORS = {
        "mm": 0.001,
        "cm": 0.01,
        "m": 1.0
    }

    @classmethod
    def parse(
        cls,
        text: str
    ):

        match = cls.PATTERN.search(
            text
        )

        if not match:

            return None

        value = float(
            match.group("value")
            .replace(",", ".")
        )

        unit = match.group(
            "unit"
        )

        if unit is None:

            unit = "m"

        unit = unit.lower()

        return (
            value,
            unit,
            value * cls.FACTORS[unit]
        )


# ============================================================
# DIMENSION LINE DETECTOR
# ============================================================

class DimensionLineDetector:

    def __init__(
        self,
        config: Config
    ):

        self.config = config

        self.parser = (
            DimensionParser()
        )

    def detect(
        self,
        lines: list[Segment2D],
        texts: list[TextObject],
        catalog: ObjectCatalog
    ):

        dimensions = []

        for text in texts:

            parsed = self.parser.parse(
                text.text
            )

            if parsed is None:

                continue

            value, unit, value_m = parsed

            center = (
                text.bbox.center
            )

            candidates = []

            for line in lines:

                if (
                    line.length
                    <
                    self.config.DIMENSION_MIN_LENGTH
                ):

                    continue

                line_distance = (
                    GeometryMath
                    .point_segment_distance(
                        center,
                        line
                    )
                )

                if (
                    line_distance
                    <=
                    self.config.DIMENSION_MAX_TEXT_DISTANCE
                ):

                    candidates.append(
                        (
                            line_distance,
                            line
                        )
                    )

            if not candidates:

                continue

            candidates.sort(
                key=lambda x: x[0]
            )

            line_distance, line = (
                candidates[0]
            )

            if (
                GeometryMath.angle_difference(
                    line.angle_deg,
                    0
                )
                <=
                self.config.DIMENSION_ANGLE_TOLERANCE
            ):

                orientation = "horizontal"
                dimension_type = "width"

            elif (
                GeometryMath.angle_difference(
                    line.angle_deg,
                    90
                )
                <=
                self.config.DIMENSION_ANGLE_TOLERANCE
            ):

                orientation = "vertical"
                dimension_type = "height"

            else:

                orientation = "diagonal"
                dimension_type = "length"

            confidence = max(
                0.0,
                min(
                    100.0,
                    text.confidence
                )
            )

            dimension = DimensionObject(

                id=catalog.new_id(
                    "dimension"
                ),

                value=value,

                unit=unit,

                value_m=value_m,

                bbox=text.bbox,

                dimension_line=line,

                orientation=orientation,

                dimension_type=dimension_type,

                endpoint1=line.p1,

                endpoint2=line.p2,

                confidence=confidence
            )

            catalog.add_dimension(
                dimension
            )

            dimensions.append(
                dimension
            )

        return dimensions


# ============================================================
# DIMENSION REFERENCE RESOLVER
# ============================================================

class DimensionReferenceResolver:

    def __init__(
        self,
        config: Config,
        catalog: ObjectCatalog
    ):

        self.config = config
        self.catalog = catalog

    def resolve(self):

        for dimension in (
            self.catalog.dimensions.values()
        ):

            if (
                dimension.endpoint1 is None
                or
                dimension.endpoint2 is None
            ):

                continue

            candidate1 = self.find_candidates(
                dimension.endpoint1
            )

            candidate2 = self.find_candidates(
                dimension.endpoint2
            )

            pair = self.select_best_pair(
                dimension,
                candidate1,
                candidate2
            )

            if pair is None:

                continue

            obj1, projection1, score1 = (
                pair[0]
            )

            obj2, projection2, score2 = (
                pair[1]
            )

            dimension.reference1 = (
                obj1.id
            )

            dimension.reference2 = (
                obj2.id
            )

            self.catalog.add_reference(
                dimension.id,
                obj1.id
            )

            if obj2.id != obj1.id:

                self.catalog.add_reference(
                    dimension.id,
                    obj2.id
                )

            self.apply_dimension(
                dimension,
                obj1,
                obj2
            )

    def find_candidates(
        self,
        endpoint: Point
    ):

        result = []

        for obj in (
            self.catalog.objects.values()
        ):

            if obj.bbox is None:

                continue

            best_distance = float(
                "inf"
            )

            best_projection = None

            if len(obj.geometry) >= 2:

                for i in range(
                    len(obj.geometry)
                ):

                    a = obj.geometry[i]

                    b = obj.geometry[
                        (i + 1)
                        %
                        len(obj.geometry)
                    ]

                    segment = Segment2D(
                        a,
                        b
                    )

                    current_distance = (
                        GeometryMath
                        .point_segment_distance(
                            endpoint,
                            segment
                        )
                    )

                    if (
                        current_distance
                        <
                        best_distance
                    ):

                        best_distance = (
                            current_distance
                        )

                        projection, _ = (
                            GeometryMath
                            .project_point(
                                endpoint,
                                segment
                            )
                        )

                        best_projection = (
                            projection
                        )

            if (
                best_distance
                <=
                self.config.DIMENSION_OBJECT_DISTANCE
            ):

                result.append(
                    (
                        obj,
                        best_projection,
                        best_distance
                    )
                )

        result.sort(
            key=lambda x: x[2]
        )

        return result[:10]

    def select_best_pair(
        self,
        dimension,
        candidates1,
        candidates2
    ):

        if (
            not candidates1
            or
            not candidates2
        ):

            return None

        best = None
        best_score = float(
            "inf"
        )

        dimension_length = (
            dimension.dimension_line.length
        )

        for c1 in candidates1:

            for c2 in candidates2:

                obj1 = c1[0]
                obj2 = c2[0]

                score = (
                    c1[2]
                    +
                    c2[2]
                )

                if obj1.id == obj2.id:

                    score *= 0.8

                p1 = c1[1]
                p2 = c2[1]

                if p1 is not None and p2 is not None:

                    geometric_length = (
                        p1.distance(p2)
                    )

                    if geometric_length > 0:

                        ratio = (
                            geometric_length
                            /
                            max(
                                dimension_length,
                                1e-9
                            )
                        )

                        score += (
                            abs(
                                math.log(
                                    max(
                                        ratio,
                                        1e-9
                                    )
                                )
                            )
                            * 20
                        )

                if score < best_score:

                    best_score = score

                    best = (
                        c1,
                        c2
                    )

        return best

    @staticmethod
    def apply_dimension(
        dimension,
        obj1,
        obj2
    ):

        value = dimension.value_m

        if (
            dimension.dimension_type
            ==
            "width"
        ):

            if obj1.id == obj2.id:

                obj1.width_m = value

            else:

                for obj in (
                    obj1,
                    obj2
                ):

                    if (
                        obj.ifc_type
                        ==
                        IFCType.IFC_SPACE
                    ):

                        obj.width_m = value

        elif (
            dimension.dimension_type
            ==
            "height"
        ):

            if obj1.id == obj2.id:

                obj1.height_m = value

            else:

                for obj in (
                    obj1,
                    obj2
                ):

                    if (
                        obj.ifc_type
                        ==
                        IFCType.IFC_SPACE
                    ):

                        obj.height_m = value

        elif (
            dimension.dimension_type
            ==
            "length"
        ):

            if obj1.id == obj2.id:

                obj1.length_m = value


# ============================================================
# SCALE ESTIMATOR
# ============================================================

class ScaleEstimator:

    def __init__(self):

        self.meters_per_pixel = None
        self.source_dimension_id = None

    def estimate(
        self,
        catalog: ObjectCatalog
    ):

        samples = []

        for dimension in (
            catalog.dimensions.values()
        ):

            line_length = (
                dimension.dimension_line.length
            )

            if line_length <= 0:

                continue

            scale = (
                dimension.value_m
                /
                line_length
            )

            samples.append(
                (
                    scale,
                    dimension.id
                )
            )

        if not samples:

            return None

        scales = [
            x[0]
            for x in samples
        ]

        self.meters_per_pixel = float(
            np.median(scales)
        )

        best = min(
            samples,
            key=lambda x:
                abs(
                    x[0]
                    -
                    self.meters_per_pixel
                )
        )

        self.source_dimension_id = (
            best[1]
        )

        return (
            self.meters_per_pixel
        )

    def apply(
        self,
        catalog: ObjectCatalog
    ):

        if (
            self.meters_per_pixel
            is None
        ):

            return

        scale = (
            self.meters_per_pixel
        )

        for obj in (
            catalog.objects.values()
        ):

            if obj.width_m is None:

                if obj.width_px > 0:

                    obj.width_m = (
                        obj.width_px
                        * scale
                    )

            if obj.height_m is None:

                if obj.height_px > 0:

                    obj.height_m = (
                        obj.height_px
                        * scale
                    )

            if obj.length_m is None:

                if obj.length_px > 0:

                    obj.length_m = (
                        obj.length_px
                        * scale
                    )

            if obj.area_m2 is None:

                if obj.area_px2 > 0:

                    obj.area_m2 = (
                        obj.area_px2
                        * scale
                        * scale
                    )


# ============================================================
# IFC SEMANTIC CLASSIFIER
# ============================================================

class IFCSemanticClassifier:

    def classify_text(
        self,
        catalog: ObjectCatalog
    ):

        for text in catalog.texts.values():

            semantic = (
                IFCTextClassifier.classify(
                    text.text
                )
            )

            if semantic is None:

                continue

            candidates = []

            center = (
                text.bbox.center
            )

            for obj in catalog.objects.values():

                if obj.bbox is None:

                    continue

                current_distance = (
                    self.bbox_distance(
                        center,
                        obj.bbox
                    )
                )

                if current_distance < 120:

                    candidates.append(
                        (
                            current_distance,
                            obj
                        )
                    )

            candidates.sort(
                key=lambda x: x[0]
            )

            if not candidates:

                continue

            obj = candidates[0][1]

            if (
                obj.ifc_type
                ==
                IFCType.IFC_BUILDING_ELEMENT_PROXY
            ):

                obj.ifc_type = semantic

            if obj.id not in text.references:

                text.references.append(
                    obj.id
                )

            if text.id not in obj.referenced_by:

                obj.referenced_by.append(
                    text.id
                )

    @staticmethod
    def bbox_distance(
        point: Point,
        bbox: BoundingBox
    ):

        dx = max(
            bbox.x - point.x,
            0,
            point.x - bbox.x2
        )

        dy = max(
            bbox.y - point.y,
            0,
            point.y - bbox.y2
        )

        return math.hypot(
            dx,
            dy
        )


# ============================================================
# OBJECT MEASUREMENT CALCULATOR
# ============================================================

class ObjectMeasurementCalculator:

    """
    Calculates geometric measurements.

    Existing semantic measurements are preserved where they
    have already been established, for example wall thickness
    or a dimension-derived opening width.
    """

    def calculate(
        self,
        catalog: ObjectCatalog
    ):

        for obj in catalog.objects.values():

            if len(obj.geometry) < 2:

                continue

            points = np.array(
                [
                    [p.x, p.y]
                    for p in obj.geometry
                ],
                dtype=np.float32
            )

            if len(points) >= 2:

                rect = cv2.minAreaRect(
                    points
                )

                (_, _), (a, b), angle = (
                    rect
                )

                geometric_width = float(
                    max(a, b)
                )

                geometric_height = float(
                    min(a, b)
                )

                # Only establish geometric values when
                # no semantic value exists already.
                if obj.width_px <= 0:

                    obj.width_px = (
                        geometric_width
                    )

                if obj.height_px <= 0:

                    obj.height_px = (
                        geometric_height
                    )

                if obj.angle_deg == 0.0:

                    obj.angle_deg = float(
                        angle
                    )

            if (
                obj.length_px <= 0
                and
                len(obj.geometry) == 2
            ):

                obj.length_px = (
                    obj.geometry[0]
                    .distance(
                        obj.geometry[1]
                    )
                )

            if (
                obj.area_px2 <= 0
                and
                len(obj.geometry) >= 3
            ):

                obj.area_px2 = (
                    GeometryMath
                    .polygon_area(
                        obj.geometry
                    )
                )

            obj.properties.width = (
                obj.width_m
            )

            obj.properties.height = (
                obj.height_m
            )

            obj.properties.length = (
                obj.length_m
            )

            obj.properties.area = (
                obj.area_m2
            )


# ============================================================
# OPENING DETECTOR
# ============================================================

class OpeningDetector:

    def __init__(
        self,
        config: Config,
        catalog: ObjectCatalog
    ):

        self.config = config
        self.catalog = catalog

        self.factory = IFCObjectFactory(
            catalog
        )

    def detect(
        self,
        gaps: list[Gap]
    ):

        for gap in gaps:

            if not (
                self.config.MIN_OPENING_GAP
                <= gap.length
                <= self.config.MAX_OPENING_GAP
            ):

                continue

            angle = math.radians(
                gap.line.angle_deg
            )

            ux = math.cos(angle)
            uy = math.sin(angle)

            # gap.start/end are projections relative to
            # gap.line.p1.
            p1 = Point(
                gap.line.p1.x
                + ux * gap.start,
                gap.line.p1.y
                + uy * gap.start
            )

            p2 = Point(
                gap.line.p1.x
                + ux * gap.end,
                gap.line.p1.y
                + uy * gap.end
            )

            obj = self.factory.create(
                IFCType.IFC_OPENING_ELEMENT,
                [
                    p1,
                    p2
                ],
                "Opening"
            )

            obj.width_px = (
                gap.length
            )

            obj.length_px = (
                gap.length
            )

            obj.properties = IFCProperties(
                object_type="OPENING",
                properties={
                    "gap_length_px":
                        gap.length
                }
            )


# ============================================================
# DOOR / WINDOW SEMANTIC REFINEMENT
# ============================================================

class OpeningSemanticRefiner:

    def refine(
        self,
        catalog: ObjectCatalog
    ):

        for obj in catalog.objects.values():

            if (
                obj.ifc_type
                !=
                IFCType.IFC_OPENING_ELEMENT
            ):

                continue

            if obj.bbox is None:

                continue

            center = (
                obj.bbox.center
            )

            best = None

            for text in catalog.texts.values():

                text_bbox = text.bbox

                text_center = (
                    text_bbox.center
                )

                current_distance = (
                    center.distance(
                        text_center
                    )
                )

                if (
                    best is None
                    or
                    current_distance < best[0]
                ):

                    best = (
                        current_distance,
                        text
                    )

            if best is None:

                continue

            current_distance, text = best

            if current_distance > 100:

                continue

            semantic = (
                IFCTextClassifier.classify(
                    text.text
                )
            )

            if semantic in (
                IFCType.IFC_DOOR,
                IFCType.IFC_WINDOW
            ):

                obj.ifc_type = semantic

                if text.id not in obj.referenced_by:

                    obj.referenced_by.append(
                        text.id
                    )

                if obj.id not in text.references:

                    text.references.append(
                        obj.id
                    )


# ============================================================
# REGION ASSIGNMENT
# ============================================================

class RegionAssignment:

    def assign(
        self,
        catalog: ObjectCatalog
    ):

        for obj in catalog.objects.values():

            if obj.bbox is None:

                continue

            center = (
                obj.bbox.center
            )

            best_region = None
            best_area = float("inf")

            for region in catalog.regions.values():

                if region.bbox.contains(
                    center
                ):

                    if (
                        region.area_px2
                        <
                        best_area
                    ):

                        best_area = (
                            region.area_px2
                        )

                        best_region = (
                            region
                        )

            if best_region:

                obj.segment_id = (
                    best_region.id
                )

                if (
                    obj.id
                    not in
                    best_region.geometry_ids
                ):

                    best_region.geometry_ids.append(
                        obj.id
                    )


# ============================================================
# JSON SERIALIZER
# ============================================================

class CatalogSerializer:

    @staticmethod
    def point(
        point: Optional[Point]
    ):

        if point is None:

            return None

        return {
            "x": point.x,
            "y": point.y
        }

    @staticmethod
    def bbox(
        bbox: Optional[BoundingBox]
    ):

        if bbox is None:

            return None

        return {
            "x": bbox.x,
            "y": bbox.y,
            "width": bbox.width,
            "height": bbox.height
        }

    @classmethod
    def object(
        cls,
        obj: IFCObject
    ):

        return {

            "id": obj.id,

            "ifc_type":
                obj.ifc_type.value,

            "name":
                obj.name,

            "geometry": [
                cls.point(p)
                for p in obj.geometry
            ],

            "bbox":
                cls.bbox(obj.bbox),

            "angle_deg":
                obj.angle_deg,

            "dimensions": {

                "width_px":
                    obj.width_px,

                "height_px":
                    obj.height_px,

                "length_px":
                    obj.length_px,

                "area_px2":
                    obj.area_px2,

                "width_m":
                    obj.width_m,

                "height_m":
                    obj.height_m,

                "length_m":
                    obj.length_m,

                "area_m2":
                    obj.area_m2
            },

            "properties": {

                "name":
                    obj.properties.name,

                "description":
                    obj.properties.description,

                "object_type":
                    obj.properties.object_type,

                "width":
                    obj.properties.width,

                "height":
                    obj.properties.height,

                "length":
                    obj.properties.length,

                "area":
                    obj.properties.area,

                "properties":
                    obj.properties.properties
            },

            "segment_id":
                obj.segment_id,

            "references":
                obj.references,

            "referenced_by":
                obj.referenced_by,

            "confidence":
                obj.confidence
        }

    @classmethod
    def text(
        cls,
        text: TextObject
    ):

        return {

            "id":
                text.id,

            "text":
                text.text,

            "bbox":
                cls.bbox(text.bbox),

            "confidence":
                text.confidence,

            "references":
                text.references
        }

    @classmethod
    def dimension(
        cls,
        dimension: DimensionObject
    ):

        return {

            "id":
                dimension.id,

            "value":
                dimension.value,

            "unit":
                dimension.unit,

            "value_m":
                dimension.value_m,

            "orientation":
                dimension.orientation,

            "dimension_type":
                dimension.dimension_type,

            "bbox":
                cls.bbox(
                    dimension.bbox
                ),

            "dimension_line": {

                "p1":
                    cls.point(
                        dimension.dimension_line.p1
                    ),

                "p2":
                    cls.point(
                        dimension.dimension_line.p2
                    )
            },

            "endpoint1":
                cls.point(
                    dimension.endpoint1
                ),

            "endpoint2":
                cls.point(
                    dimension.endpoint2
                ),

            "reference1":
                dimension.reference1,

            "reference2":
                dimension.reference2,

            "confidence":
                dimension.confidence
        }

    @classmethod
    def region(
        cls,
        region: PlanRegion
    ):

        return {

            "id":
                region.id,

            "bbox":
                cls.bbox(
                    region.bbox
                ),

            "area_px2":
                region.area_px2,

            "geometry_ids":
                region.geometry_ids
        }

    @classmethod
    def catalog(
        cls,
        catalog: ObjectCatalog
    ):

        return {

            "objects": [
                cls.object(x)
                for x in catalog.objects.values()
            ],

            "texts": [
                cls.text(x)
                for x in catalog.texts.values()
            ],

            "dimensions": [
                cls.dimension(x)
                for x in catalog.dimensions.values()
            ],

            "regions": [
                cls.region(x)
                for x in catalog.regions.values()
            ]
        }


# ============================================================
# VISUALIZER
# ============================================================

class Visualizer:

    def draw(
        self,
        image,
        lines,
        catalog: ObjectCatalog
    ):

        result = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

        # ----------------------------------------------------
        # Reconstructed lines
        # ----------------------------------------------------

        for line in lines:

            cv2.line(
                result,
                (
                    int(round(line.p1.x)),
                    int(round(line.p1.y))
                ),
                (
                    int(round(line.p2.x)),
                    int(round(line.p2.y))
                ),
                (255, 0, 0),
                1
            )

        # ----------------------------------------------------
        # IFC objects
        # ----------------------------------------------------

        for obj in catalog.objects.values():

            if obj.bbox is None:

                continue

            x = int(
                round(obj.bbox.x)
            )

            y = int(
                round(obj.bbox.y)
            )

            w = int(
                round(obj.bbox.width)
            )

            h = int(
                round(obj.bbox.height)
            )

            cv2.rectangle(
                result,
                (x, y),
                (
                    x + w,
                    y + h
                ),
                (0, 255, 0),
                1
            )

            label = (
                f"{obj.ifc_type.value}"
                f" {obj.id}"
            )

            cv2.putText(
                result,
                label,
                (
                    x,
                    max(
                        12,
                        y - 3
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            dimensions = []

            if obj.width_m is not None:

                dimensions.append(
                    f"W={obj.width_m:.2f}"
                )

            if obj.height_m is not None:

                dimensions.append(
                    f"H={obj.height_m:.2f}"
                )

            if dimensions:

                text = " ".join(
                    dimensions
                )

                cv2.putText(
                    result,
                    text,
                    (
                        x,
                        y + h + 12
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        for dimension in (
            catalog.dimensions.values()
        ):

            line = (
                dimension.dimension_line
            )

            cv2.line(
                result,
                (
                    int(round(line.p1.x)),
                    int(round(line.p1.y))
                ),
                (
                    int(round(line.p2.x)),
                    int(round(line.p2.y))
                ),
                (0, 255, 255),
                2
            )

            for p in (
                line.p1,
                line.p2
            ):

                cv2.circle(
                    result,
                    (
                        int(round(p.x)),
                        int(round(p.y))
                    ),
                    4,
                    (0, 0, 255),
                    -1
                )

            label = (
                f"{dimension.value:g}"
                f"{dimension.unit}"
                f" -> "
                f"{dimension.reference1},"
                f"{dimension.reference2}"
            )

            cv2.putText(
                result,
                label,
                (
                    int(round(line.midpoint.x)),
                    int(round(line.midpoint.y))
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        return result


# ============================================================
# FLOOR PLAN ANALYZER
# ============================================================

class FloorPlanAnalyzer:

    def __init__(
        self,
        config: Config
    ):

        self.config = config

        self.processor = ImageProcessor(
            config
        )

        self.line_detector = LineDetector(
            config
        )

        self.reconstructor = LineReconstructor(
            config
        )

        self.gap_detector = GapDetector(
            config
        )

        self.region_detector = (
            PlanRegionDetector(
                config
            )
        )

        self.ocr = OCRReader(
            config
        )

        self.dimension_detector = (
            DimensionLineDetector(
                config
            )
        )

        self.scale_estimator = (
            ScaleEstimator()
        )

    def analyze(
        self,
        filename: str
    ):

        print()
        print("=" * 70)
        print("IFC SEMANTIC FLOOR PLAN ANALYZER")
        print("=" * 70)

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = self.processor.load(
            filename
        )

        print(
            f"Image: "
            f"{image.shape[1]} x "
            f"{image.shape[0]}"
        )

        # ----------------------------------------------------
        # Edges
        # ----------------------------------------------------

        edges = self.processor.edges(
            image
        )

        print(
            f"Edges: "
            f"{int(np.count_nonzero(edges))}"
        )

        # ----------------------------------------------------
        # Raw lines
        # ----------------------------------------------------

        raw_lines = (
            self.line_detector.detect(
                edges
            )
        )

        print(
            f"Raw lines: "
            f"{len(raw_lines)}"
        )

        # ----------------------------------------------------
        # Reconstruct lines
        # ----------------------------------------------------

        lines = (
            self.reconstructor.reconstruct(
                raw_lines
            )
        )

        print(
            f"Reconstructed lines: "
            f"{len(lines)}"
        )

        # ----------------------------------------------------
        # Catalog
        # ----------------------------------------------------

        catalog = ObjectCatalog()

        # ----------------------------------------------------
        # Regions BEFORE detailed geometry
        # ----------------------------------------------------

        regions = (
            self.region_detector.detect(
                image.shape,
                lines,
                catalog
            )
        )

        print(
            f"Whitespace regions: "
            f"{len(regions)}"
        )

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        geometry = GeometryAnalyzer(
            self.config,
            catalog
        )

        geometry.analyze_lines(
            lines
        )

        geometry.detect_walls(
            lines
        )

        geometry.detect_spaces(
            regions
        )

        geometry.detect_stairs(
            lines
        )

        # ----------------------------------------------------
        # Gaps / openings
        # ----------------------------------------------------

        gaps = (
            self.gap_detector.detect(
                raw_lines,
                lines
            )
        )

        print(
            f"Line gaps: "
            f"{len(gaps)}"
        )

        openings = OpeningDetector(
            self.config,
            catalog
        )

        openings.detect(
            gaps
        )

        print(
            f"Opening candidates: "
            f"{sum(
                1
                for x in catalog.objects.values()
                if x.ifc_type
                == IFCType.IFC_OPENING_ELEMENT
            )}"
        )

        # ----------------------------------------------------
        # First measurement calculation
        # ----------------------------------------------------

        measurements = (
            ObjectMeasurementCalculator()
        )

        measurements.calculate(
            catalog
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        texts = self.ocr.read(
            image,
            catalog
        )

        print(
            f"OCR objects: "
            f"{len(texts)}"
        )

        # ----------------------------------------------------
        # IFC semantic classification
        # ----------------------------------------------------

        semantic = (
            IFCSemanticClassifier()
        )

        semantic.classify_text(
            catalog
        )

        # ----------------------------------------------------
        # Refine openings
        # ----------------------------------------------------

        OpeningSemanticRefiner().refine(
            catalog
        )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        dimensions = (
            self.dimension_detector.detect(
                lines,
                texts,
                catalog
            )
        )

        print(
            f"Dimensions: "
            f"{len(dimensions)}"
        )

        # ----------------------------------------------------
        # Dimension -> geometry
        # ----------------------------------------------------

        resolver = (
            DimensionReferenceResolver(
                self.config,
                catalog
            )
        )

        resolver.resolve()

        resolved_dimensions = sum(
            1
            for d in catalog.dimensions.values()
            if (
                d.reference1 is not None
                or
                d.reference2 is not None
            )
        )

        print(
            f"Resolved dimensions: "
            f"{resolved_dimensions}"
        )

        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        scale = (
            self.scale_estimator.estimate(
                catalog
            )
        )

        if scale is not None:

            print(
                f"Scale: "
                f"{scale:.8f} m/pixel"
            )

            print(
                f"Pixels/m: "
                f"{1.0 / scale:.2f}"
            )

        else:

            print(
                "Scale: unknown"
            )

        # ----------------------------------------------------
        # Apply scale
        # ----------------------------------------------------

        self.scale_estimator.apply(
            catalog
        )

        # ----------------------------------------------------
        # Recalculate properties
        # ----------------------------------------------------

        measurements.calculate(
            catalog
        )

        # ----------------------------------------------------
        # Region assignment
        # ----------------------------------------------------

        RegionAssignment().assign(
            catalog
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        self.print_summary(
            catalog
        )

        return (
            image,
            edges,
            lines,
            catalog
        )

    @staticmethod
    def print_summary(
        catalog: ObjectCatalog
    ):

        print()
        print("-" * 70)
        print("IFC OBJECT SUMMARY")
        print("-" * 70)

        counts = {}

        for obj in catalog.objects.values():

            key = obj.ifc_type.value

            counts[key] = (
                counts.get(key, 0)
                + 1
            )

        for key, count in sorted(
            counts.items()
        ):

            print(
                f"{key:35s} {count:6d}"
            )

        print()
        print("-" * 70)
        print("DIMENSIONS")
        print("-" * 70)

        for obj in catalog.objects.values():

            if (
                obj.width_m is None
                and
                obj.height_m is None
                and
                obj.length_m is None
            ):

                continue

            print(
                f"{obj.id:16s} "
                f"{obj.ifc_type.value:28s} "
                f"W={str(obj.width_m):>10} "
                f"H={str(obj.height_m):>10} "
                f"L={str(obj.length_m):>10} "
                f"A={str(obj.area_m2):>12}"
            )

        print()
        print("-" * 70)
        print("DIMENSION REFERENCES")
        print("-" * 70)

        for dimension in (
            catalog.dimensions.values()
        ):

            print(
                f"{dimension.id:20s} "
                f"{dimension.value:g}"
                f"{dimension.unit:3s} "
                f"{dimension.dimension_type:8s} "
                f"-> "
                f"{dimension.reference1} "
                f"<-> "
                f"{dimension.reference2}"
            )


# ============================================================
# APPLICATION
# ============================================================

class FloorPlanApplication:

    def __init__(
        self,
        config: Config
    ):

        self.config = config

        self.analyzer = (
            FloorPlanAnalyzer(
                config
            )
        )

        self.visualizer = (
            Visualizer()
        )

    def run(
        self,
        filename: str,
        json_output: str,
        visualization_output: Optional[str]
    ):

        image, edges, lines, catalog = (
            self.analyzer.analyze(
                filename
            )
        )

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        data = (
            CatalogSerializer.catalog(
                catalog
            )
        )

        data["source"] = {
            "image": str(
                Path(filename)
            )
        }

        data["image"] = {
            "width_px":
                int(image.shape[1]),
            "height_px":
                int(image.shape[0])
        }

        data["scale"] = {

            "meters_per_pixel":
                self.analyzer
                .scale_estimator
                .meters_per_pixel,

            "source_dimension_id":
                self.analyzer
                .scale_estimator
                .source_dimension_id
        }

        print("\nIfcOpeningElement dimensions:")

        for obj in data:
        
            if obj.ifc_type != "IfcOpeningElement":
                continue
        
            opening = Opening(
                obj=obj,
                opening_type=OpeningType.OPENING,
            )
        
            print(
                f"  {obj.id}: "
                f"width={opening.width:.2f}, "
                f"height={opening.height:.2f}, "
                f"name={obj.name!r}"
                )
                
        with open(
            json_output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

        print()
        print(
            f"Catalog: "
            f"{json_output}"
        )
        


        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        if visualization_output:

            result = (
                self.visualizer.draw(
                    image,
                    lines,
                    catalog
                )
            )

            success = cv2.imwrite(
                visualization_output,
                result
            )

            if not success:

                raise RuntimeError(
                    "Could not write visualization: "
                    f"{visualization_output}"
                )

            print(
                f"Visualization: "
                f"{visualization_output}"
            )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "IFC semantic floor plan analyzer"
        )
    )

    parser.add_argument(
        "image",
        help="Input technical drawing"
    )

    parser.add_argument(
        "--output",
        default="catalog.json",
        help="JSON catalog"
    )

    parser.add_argument(
        "--visualization",
        default="floorplan_result.png",
        help="Visualization"
    )

    args = parser.parse_args()

    application = (
        FloorPlanApplication(
            Config()
        )
    )

    application.run(
        args.image,
        args.output,
        args.visualization
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
