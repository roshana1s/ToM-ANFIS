class ToMFeatureExtractor:
    """
    Extract 10 Theory of Mind (ToM) cognitive features from the PIE database.

    Output columns per pedestrian:
        Metadata (7) : set_id, video_id, pedestrian_id, prediction_frame,
                       crossing_frame, frames_to_crossing, crossing_label
        Belief   (4) : B1, B2, B3, B4
        Goal     (2) : G1 (target), G2
        Attention(2) : A1, A2
        Context  (2) : C1, C2
    """

    def __init__(self, db: Dict, observation_window: int = 15):
        """
        Args:
            db               : PIE database dict loaded from pie_database.pkl
            observation_window: Number of frames to aggregate temporal features over
        """
        self.db       = db
        self.obs_window = observation_window

    # ── Shared utility helpers ─────────────────────────────────────────────

    def _bbox_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _bbox_dist(self, b1, b2):
        c1, c2 = self._bbox_center(b1), self._bbox_center(b2)
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

    def _safe_mean(self, vals):
        return float(np.mean(vals)) if vals else 0.0

    # ── Belief State Features ─────────────────────────────────────────────

    def _B1(self, attributes: Dict) -> float:
        """
        B1: Age Vulnerability Awareness  ∈ [0.2, 0.8]  |  static
        Children and seniors have higher self-perceived vulnerability,
        making them more cautious (higher belief value) when crossing.
        """
        return {0: 0.8, 1: 0.3, 2: 0.2, 3: 0.7}.get(attributes.get('age', 2), 0.2)

    def _B2(self, attributes: Dict) -> float:
        """
        B2: Intersection Complexity  ∈ [0.2, 1.0]  |  static
        More complex intersections impose higher cognitive load on the pedestrian,
        requiring more careful observation before deciding to cross.
        """
        return {0: 0.2, 1: 0.5, 2: 0.6, 3: 0.6, 4: 1.0}.get(
            attributes.get('intersection', 0), 0.2)

    def _B3(self, ped_bbox, traffic_data, frame_id) -> float:
        """
        B3: Position Proximity Awareness  ∈ [0.0, 1.0]  |  frame-level
        Euclidean distance from pedestrian centre to the nearest crosswalk bbox centre,
        normalised by image width (1920px) and inverted so closer = higher value.
        Returns 0.0 if no crosswalk is annotated in the current frame.
        """
        try:
            min_d = float('inf')
            for obj in traffic_data.values():
                if obj['obj_class'] == 'crosswalk' and frame_id in obj['frames']:
                    cw_bbox = obj['bbox'][obj['frames'].index(frame_id)]
                    min_d = min(min_d, self._bbox_dist(ped_bbox, cw_bbox))
            return max(0.0, 1.0 - min(min_d / 1920.0, 1.0)) if min_d < float('inf') else 0.0
        except:
            return 0.0

    def _B4(self, traffic_data, frame_ids) -> float:
        """
        B4: Vehicle Density  ∈ [0.0, 1.0]  |  temporal
        Average number of annotated vehicles per frame over the observation window,
        capped at 1.0 using a normalisation factor of 10 vehicles.
        """
        try:
            counts = [
                sum(1 for o in traffic_data.values()
                    if o['obj_class'] == 'vehicle' and fid in o['frames'])
                for fid in frame_ids
            ]
            return min(self._safe_mean(counts) / 10.0, 1.0)
        except:
            return 0.0

    # ── Goal State Features ───────────────────────────────────────────────

    def _G1(self, attributes: Dict) -> float:
        """
        G1: Crossing Intention Strength  ∈ [0.0, 1.0]  |  static  ← TARGET
        Ground-truth intention probability from PIE's crowdsourced human annotations.
        This is the variable the ANFIS model is trained to predict.
        """
        return attributes.get('intention_prob', 0.5)

    def _G2(self, ped_data, frame_indices) -> float:
        """
        G2: Urgency / Rushing Behaviour  ∈ [0.0, 1.0]  |  temporal
        Average of two sub-signals over the observation window:
          - Walking ratio      : fraction of frames where action==1 (walking)
          - Urgent gesture ratio: fraction with gesture ∈ {1,2,3,4}
            (hand acknowledgment=1, hand yield=2, right-of-way=3, nod=4)
        """
        try:
            actions = [ped_data['behavior']['action'][i]  for i in frame_indices]
            gestures = [ped_data['behavior']['gesture'][i] for i in frame_indices]
            w = sum(a == 1         for a in actions)  / len(actions)
            g = sum(g in [1,2,3,4] for g in gestures) / len(gestures)
            return (w + g) / 2.0
        except:
            return 0.0

    # ── Attention State Features ──────────────────────────────────────────

    def _A1(self, ped_data, frame_indices) -> float:
        """
        A1: Vehicle Awareness  ∈ [0.0, 1.0]  |  temporal
        Proportion of observation frames in which the pedestrian gaze is
        directed toward the ego-vehicle (PIE 'look' annotation == 1).
        """
        try:
            return self._safe_mean([ped_data['behavior']['look'][i] for i in frame_indices])
        except:
            return 0.0

    def _A2(self, ped_data, traffic_data, frame_indices) -> float:
        """
        A2: Traffic Signal Awareness  ∈ [−1.0, 1.0]  |  temporal
        For each frame, finds visible traffic lights and maps their state to a
        pedestrian-safety score, then averages across the observation window.

        Signal state encoding:
          Pedestrian lights : green(3)=1.0 safe | yellow(2)=0.5 | red(1)=0.0 unsafe
          Vehicle lights    : red(1)=1.0 safe   | yellow(2)=0.5 | green(3)=0.0 unsafe
          Transit lights    : 0.0  |  No signal at all: −1.0
        """
        try:
            scores = []
            for idx in frame_indices:
                fid = ped_data['frames'][idx]
                for obj in traffic_data.values():
                    if obj.get('obj_class') == 'traffic_light' and fid in obj.get('frames', []):
                        state = obj['state'][obj['frames'].index(fid)]
                        tl_type = obj.get('obj_type', 'regular')
                        if tl_type == 'pedestrian': v = {0:-1, 1:0.0, 2:0.5, 3:1.0}.get(state,-1)
                        elif tl_type == 'transit':    v = 0.0
                        else:                         v = {0:-1, 1:1.0, 2:0.5, 3:0.0}.get(state,-1)
                        if v >= 0:
                            scores.append(v)
            return self._safe_mean(scores) if scores else -1.0
        except:
            return -1.0

    # ── Context / Environment Features ────────────────────────────────────

    def _C1(self, vehicle_data, frame_ids) -> float:
        """
        C1: Relative Vehicle Speed  ∈ [0.0, 1.0]  |  temporal
        Mean ego-vehicle OBD speed over the observation window, normalised
        by 27 m/s (≈100 km/h) which is the typical urban speed limit ceiling.
        Captures how threatening the traffic environment feels to the pedestrian.
        """
        try:
            speeds = [vehicle_data[fid].get('OBD_speed', 0.0) for fid in frame_ids if fid in vehicle_data]
            return min(np.mean(speeds) / 27.0, 1.0) if speeds else 0.0
        except:
            return 0.0

    def _C2(self, ped_data, frame_indices) -> float:
        """
        C2: Occlusion Factor  ∈ [0.0, 1.0]  |  temporal
        Average PIE occlusion annotation over the observation window,
        normalised from {0, 1, 2} to [0, 1] by dividing by 2.
        High occlusion limits both the pedestrian's and the AV's visibility.
        """
        try:
            return self._safe_mean([ped_data['occlusion'][i] for i in frame_indices]) / 2.0
        except:
            return 0.0

    # ── Main Extraction ───────────────────────────────────────────────────

    def extract_for_pedestrian(self, set_id, vid, ped_id, prediction_horizon=30):
        """
        Extract all 10 ToM features for a single pedestrian.

        Returns:
            dict with 7 metadata columns + 10 feature values, or None if
            the observation window cannot be constructed for this clip.
        """
        try:
            video_data = self.db[set_id][vid]
            ped_data = video_data['ped_annotations'][ped_id]
            traffic_data = video_data['traffic_annotations']
            vehicle_data = video_data['vehicle_annotations']
            attributes = ped_data['attributes']
            frames = ped_data['frames']

            # Locate crossing event and define temporal windows
            cp = attributes.get('crossing_point')
            if cp is None or cp not in frames:
                cp = frames[-1]
            crossing_idx = frames.index(cp)
            prediction_idx = max(0, crossing_idx - prediction_horizon)
            start_idx = max(0, prediction_idx - self.obs_window)

            if start_idx >= prediction_idx or prediction_idx > len(frames):
                return None

            frame_indices = list(range(start_idx, prediction_idx))
            current_idx = frame_indices[-1]
            current_frame = frames[current_idx]
            current_bbox = ped_data['bbox'][current_idx]
            frame_ids = [frames[i] for i in frame_indices]

            return {
                # Metadata
                'set_id': set_id, 'video_id': vid, 'pedestrian_id': ped_id,
                'prediction_frame': current_frame, 'crossing_frame': cp,
                'frames_to_crossing': crossing_idx - current_idx,
                'crossing_label': int(attributes.get('crossing', -1) == 1),
                # Belief
                'B1_age_vulnerability_awareness':   self._B1(attributes),
                'B2_intersection_complexity':       self._B2(attributes),
                'B3_pedestrian_position_proximity': self._B3(current_bbox, traffic_data, current_frame),
                'B4_vehicle_density':               self._B4(traffic_data, frame_ids),
                # Goal
                'G1_crossing_intention_strength':   self._G1(attributes),   # ← TARGET
                'G2_urgency_rushing':               self._G2(ped_data, frame_indices),
                # Attention
                'A1_vehicle_awareness':             self._A1(ped_data, frame_indices),
                'A2_traffic_signal_awareness':      self._A2(ped_data, traffic_data, frame_indices),
                # Context
                'C1_relative_vehicle_speed':        self._C1(vehicle_data, frame_ids),
                'C2_occlusion_factor':              self._C2(ped_data, frame_indices),
            }
        except Exception as e:
            print(f'  Skipping {set_id}/{vid}/{ped_id}: {e}')
            return None

    def extract_all(self, output_csv='tom_cognitive_features.csv',
                    prediction_horizon=30) -> pd.DataFrame:
        """
        Iterate over the full PIE database and extract features for every pedestrian.

        Args:
            output_csv         : Filename to save the resulting feature CSV.
            prediction_horizon : Frames before crossing to make the prediction.

        Returns:
            pd.DataFrame  (N_pedestrians × 17 columns)
        """
        all_rows, total, ok = [], 0, 0

        print('='*65)
        print('EXTRACTING ToM FEATURES FROM PIE DATABASE')
        print(f'  Observation window : {self.obs_window} frames')
        print(f'  Prediction horizon : {prediction_horizon} frames')
        print('='*65)

        for set_id in sorted(self.db):
            for vid in sorted(self.db[set_id]):
                for ped_id in sorted(self.db[set_id][vid]['ped_annotations']):
                    total += 1
                    row = self.extract_for_pedestrian(set_id, vid, ped_id, prediction_horizon)
                    if row:
                        all_rows.append(row)
                        ok += 1
                        if ok % 200 == 0:
                            print(f'  ✓ {ok}/{total} processed...')

        if not all_rows:
            print('No features extracted — verify database structure.')
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df.to_csv(output_csv, index=False)

        success_rate = 100 * ok / total
        print(f'\n  Pedestrians total  : {total}')
        print(f'  Successfully done  : {ok}  ({success_rate:.1f}%)')
        print(f'  Output CSV         : {output_csv}')
        print(f'  DataFrame shape    : {df.shape}')

        # Flag any near-zero variance features (could indicate extraction issues)
        feat_cols = [c for c in df.columns if c[0] in ('B','G','A','C')]
        low_var = [c for c in feat_cols if df[c].std() < 0.01]
        if low_var:
            print(f'\n  Low-variance features detected: {low_var}')
        else:
            print(f'\n  All {len(feat_cols)} features have healthy variance')

        return df