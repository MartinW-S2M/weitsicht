.. _geo-ref-basic:

Geo-referenced Images
=====================

Definition
----------
A geo-referenced image is an image whose pixels are tied to real-world coordinates through a known sensor pose
(position and orientation) and camera model. Each pixel can be mapped to a ground location in a chosen spatial
reference system (e.g., WGS84, UTM), enabling measurement, overlay with GIS layers, and integration into mapping
pipelines.

Where They Come From
--------------------
- Inertial Navigation Systems (INS: IMU + GNSS): Cameras paired with IMU/GNSS log position/attitude at capture time;
  the exterior orientation is attached to each frame.
- Bundle blocks / photogrammetry rigs: Overlapping images from calibrated cameras are processed in a bundle adjustment
  to solve exterior orientations and sparse 3D structure.
- Drones / aerial vehicles: Flight controllers provide GNSS + IMU; trigger times are synced to images; optional onboard
  RTK/PPK improves accuracy.
- Crew-flown aircraft: Similar to drones but often with multi-camera nadir/oblique setups and higher-grade INS.
- Computer vision pipelines: SLAM/VO systems estimate camera trajectories; when anchored with GNSS or surveyed control,
  the trajectory becomes geo-referenced.

Information Needed for a Fully Geo-referenced Image
---------------------------------------------------
To compute a reliable mapping from pixels to ground coordinates you need:

- Exterior orientation: camera position (X, Y, Z) and attitude (roll, pitch, yaw) in a known datum/CRS at the exposure
  time.
- Interior orientation (camera intrinsics): focal length, principal point, lens distortion, sensor format; ideally
  from a calibrated camera model.
- Timing alignment: precise synchronization between shutter time and navigation data (GNSS/IMU or SLAM trajectory).
- Coordinate reference definition: horizontal and vertical datums, projection (e.g., EPSG code), and any geoid model
  used for orthometric heights.
- Metadata linkage: image filename <-> navigation record, plus quality flags (fix type, PDOP, lever-arm applied?).

Camera Calibration
------------------
- Pre-calibration (lab/field): The camera is calibrated before the mission using targets or checkerboards to solve
  intrinsics and lens distortion; results are stable if focus and temperature are controlled.
- On-the-job self-calibration: Calibration parameters are estimated during bundle adjustment/SLAM from the mission
  imagery itself; reduces bias from changing focus/temperature but requires good network geometry (overlap, roll/yaw
  variation, crossed flight lines).
- Hybrid: Start from a lab prior and allow limited adjustment in processing to absorb small in-flight changes.

Lever Arms and Boresight
------------------------
- Lever arm: 3D offset between the GNSS/IMU reference point and the camera center. Must be measured (survey) and
  applied.
- Boresight: Rotation between IMU body frame and camera frame; estimated via calibration flights or target campaigns.

Accuracy Considerations
-----------------------
- GNSS quality: Single-frequency vs RTK/PPK vs PPP; update rate; multipath environment.
- IMU grade: MEMS vs tactical; affects short-term attitude accuracy between GNSS updates.
- Flight geometry: Forward/side overlap, cross strips, varying heights/attitudes improve bundle strength.
- Timing: Sub-millisecond trigger tagging reduces position drift in fast motion.
- Ground control: Well-distributed GCPs/Check points anchor the solution and validate absolute accuracy.
- DEM/height model: Needed for orthorectification and footprint projection.

Typical Outputs
---------------
- Geo-tagged images (EXIF with GPS tags plus custom fields for attitude).
- Exterior orientation tables (CSV/JSON) per image.
- Orthophotos.

Practical Tips
--------------
- Keep a metadata manifest: image id, timestamp, GNSS fix type, PDOP, attitude, lever-arm applied?, boresight version.
- Record firmware and calibration versions with the dataset.
