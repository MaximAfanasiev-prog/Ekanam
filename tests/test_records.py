from street_falcon_reid.records import VehicleRecord, make_identity_split


def _records() -> list[VehicleRecord]:
    records: list[VehicleRecord] = []
    for identity in range(12):
        for camera in range(2):
            for frame in range(2):
                records.append(
                    VehicleRecord(
                        image_id=f"{identity}-{camera}-{frame}",
                        x=0,
                        y=0,
                        w=10,
                        h=10,
                        vehicle_id=str(identity),
                        camera_id=str(camera),
                    )
                )
    return records


def test_identity_split_is_disjoint_and_contains_open_set() -> None:
    split = make_identity_split(
        _records(),
        seed=7,
        validation_fraction=0.5,
        open_set_fraction=0.33,
    )
    train_ids = {item.vehicle_id for item in split.train}
    query_ids = {item.vehicle_id for item in split.query}
    gallery_ids = {item.vehicle_id for item in split.gallery}

    assert train_ids.isdisjoint(query_ids)
    assert set(split.open_set_validation_ids).isdisjoint(gallery_ids)
    assert set(split.matched_validation_ids).issubset(gallery_ids)
    for query in split.query:
        if query.vehicle_id in split.matched_validation_ids:
            assert any(
                item.vehicle_id == query.vehicle_id and item.camera_id != query.camera_id
                for item in split.gallery
            )
