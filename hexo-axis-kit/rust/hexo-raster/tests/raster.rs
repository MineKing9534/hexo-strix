use hexo_engine::{GameConfig, GameState, Player};
use hexo_raster::{
    build_raster, raster_dimensions, Plane, RasterBatch, RasterSpec, Scalar, NUM_PLANES,
    NUM_SCALARS,
};

fn test_spec(prune_empty_edges: bool) -> RasterSpec {
    RasterSpec {
        buckets: vec![(9, 9), (17, 17), (25, 25), (33, 33)],
        prune_empty_edges,
        ..RasterSpec::default()
    }
}

#[test]
fn full_hexo_opening_fits_17_and_preserves_legal_order() {
    let game = GameState::new();
    let raster = build_raster(&game, &test_spec(true)).unwrap();
    assert_eq!(
        raster_dimensions(&game, &test_spec(true)).unwrap(),
        (raster.width, raster.height)
    );
    assert_eq!((raster.width, raster.height), (17, 17));
    assert_eq!(raster.planes.len(), NUM_PLANES * 17 * 17);
    assert_eq!(raster.scalars.len(), NUM_SCALARS);
    assert_eq!(raster.legal_coords, game.legal_moves());
    assert_eq!(raster.legal_coords.len(), 216);
    assert_eq!(
        raster
            .active_mask
            .iter()
            .map(|&x| usize::from(x))
            .sum::<usize>(),
        217
    );

    // P2 moves first, so the opening P1 stone is an opponent stone.
    let origin_idx = raster.coord_to_index((0, 0)).unwrap();
    assert_eq!(raster.plane(Plane::OwnStone)[origin_idx], 0.0);
    assert_eq!(raster.plane(Plane::OppStone)[origin_idx], 1.0);
    assert_eq!(raster.scalars[Scalar::MovesRemaining.index()], 1.0);
    assert_eq!(raster.scalars[Scalar::ToMoveIdentity.index()], -1.0);

    for (&coord, &flat) in raster
        .legal_coords
        .iter()
        .zip(raster.legal_flat_indices.iter())
    {
        assert_eq!(raster.index_to_coord(flat as usize), Some(coord));
        assert_eq!(raster.plane(Plane::Legal)[flat as usize], 1.0);
    }
}

#[test]
fn blocker_aware_ray_mask_matches_axis_walk_semantics() {
    let config = GameConfig {
        win_length: 6,
        placement_radius: 2,
        max_moves: 100,
    };
    let stones = [((0, 0), Player::P1), ((2, 0), Player::P2)];
    let game = GameState::from_state(&stones, Player::P1, 2, config);
    let raster = build_raster(&game, &test_spec(true)).unwrap();
    let origin = raster.coord_to_index((0, 0)).unwrap();

    // Ray 0 is +q. The opponent at distance 2 is connected, then stops the
    // P1-source walk. The legal empty at distance 3 therefore is not connected.
    assert!(raster.has_ray_source(origin, 0, 2));
    assert!(!raster.has_ray_source(origin, 0, 3));

    // The mask is symmetric: at (2,0), the opening stone arrives from ray 3.
    let blocker = raster.coord_to_index((2, 0)).unwrap();
    assert!(raster.has_ray_source(blocker, 3, 2));
}

#[test]
fn empty_to_empty_edges_obey_pruning_flag() {
    let config = GameConfig {
        win_length: 6,
        placement_radius: 2,
        max_moves: 100,
    };
    let stones = [((0, 0), Player::P1), ((2, 0), Player::P2)];
    let game = GameState::from_state(&stones, Player::P1, 2, config);

    let pruned = build_raster(&game, &test_spec(true)).unwrap();
    let dense = build_raster(&game, &test_spec(false)).unwrap();
    let a = pruned.coord_to_index((3, 0)).unwrap();
    let b = pruned.coord_to_index((4, 0)).unwrap();
    assert_eq!(pruned.index_to_coord(a), Some((3, 0)));
    assert_eq!(pruned.index_to_coord(b), Some((4, 0)));
    assert!(!pruned.has_ray_source(a, 0, 1));
    assert!(dense.has_ray_source(a, 0, 1));
}

#[test]
fn same_shape_positions_pack_and_encode() {
    let a = build_raster(&GameState::new(), &test_spec(true)).unwrap();
    let b = build_raster(&GameState::new(), &test_spec(true)).unwrap();
    let batch = RasterBatch::from_positions(&[a, b]).unwrap();
    assert_eq!(batch.batch_size, 2);
    assert_eq!(batch.legal_offsets, vec![0, 216, 432]);
    assert_eq!(batch.legal_flat_indices.len(), 432);
    let bytes = batch.encode_hxr1();
    assert_eq!(&bytes[0..4], b"HXR1");
}

#[test]
fn mixed_shape_positions_are_padded_without_reordering_actions() {
    let small = build_raster(
        &GameState::with_config(GameConfig {
            win_length: 4,
            placement_radius: 2,
            max_moves: 30,
        }),
        &test_spec(true),
    )
    .unwrap();
    let large = build_raster(&GameState::new(), &test_spec(true)).unwrap();
    assert_ne!((small.width, small.height), (large.width, large.height));

    let expected = [small.legal_coords.clone(), large.legal_coords.clone()];
    let batch = RasterBatch::from_positions(&[small, large]).unwrap();
    assert_eq!((batch.width, batch.height), (17, 17));

    for (batch_index, expected) in expected.iter().enumerate() {
        let start = batch.legal_offsets[batch_index] as usize;
        let end = batch.legal_offsets[batch_index + 1] as usize;
        let origin = batch.origins[batch_index];
        let actual = batch.legal_flat_indices[start..end]
            .iter()
            .map(|&global| {
                let local = global as usize - batch_index * 17 * 17;
                let y = local / 17;
                let x = local % 17;
                (origin.0 + x as i32, origin.1 + y as i32)
            })
            .collect::<Vec<_>>();
        assert_eq!(&actual, expected);
    }
}
