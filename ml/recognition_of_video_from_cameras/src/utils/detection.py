from typing import Dict, Iterable, List, Set, Any
import numpy as np
import supervision as sv


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}     # сопоставление tracker_id с индексом zone_in в PolygonHandler.zones_in
        self.counts: Dict[int, Dict[int, Set[int]]] = {}    # словарь содержащий детекции по индексу zone_in_id

    def update(
            self,
            detections_all: sv.Detections,
            detections_in_zones: List[sv.Detections],
            detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        # задание словаря сопоставления self.tracker_id_to_zone_id
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        # задание детекций с каждым фреймом
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]
