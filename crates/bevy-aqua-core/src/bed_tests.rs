use super::*;
use bevy::render::MainWorld;

#[test]
fn extraction_tracks_insertion_replacement_and_removal() {
    let mut main = MainWorld::default();
    let mut images = Assets::<Image>::default();
    let a = BedHeightMap::from_height_fn(&mut images, |x, _| x, 2, Vec2::ZERO, 1.0);
    let b = BedHeightMap::from_height_fn(&mut images, |x, _| -x, 2, Vec2::splat(4.0), 2.0);
    main.insert_resource(a.clone());
    let mut render = World::new();
    render.insert_resource(main);
    let mut extract = Schedule::new(ExtractSchedule);
    extract.add_systems(extract_bed);
    extract.run(&mut render);
    assert_eq!(render.resource::<BedHeightMap>().image, a.image);
    assert_eq!(
        render.resource::<BedHeightMap>().height_range,
        a.height_range
    );

    render
        .resource_mut::<MainWorld>()
        .insert_resource(b.clone());
    extract.run(&mut render);
    assert_eq!(render.resource::<BedHeightMap>().image, b.image);
    assert_eq!(render.resource::<BedHeightMap>().origin, b.origin);
    assert_eq!(render.resource::<BedHeightMap>().size, b.size);
    assert_eq!(
        render.resource::<BedHeightMap>().height_range,
        b.height_range
    );

    render
        .resource_mut::<MainWorld>()
        .remove_resource::<BedHeightMap>();
    extract.run(&mut render);
    assert!(!render.contains_resource::<BedHeightMap>());
    extract.run(&mut render);
    assert!(!render.contains_resource::<BedHeightMap>());
    render
        .resource_mut::<MainWorld>()
        .insert_resource(a.clone());
    extract.run(&mut render);
    assert_eq!(render.resource::<BedHeightMap>().image, a.image);
}
