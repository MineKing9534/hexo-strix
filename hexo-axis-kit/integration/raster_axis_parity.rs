//! Copy to `hexo-rs/hexo-mcts/tests/raster_axis_parity.rs` after adding
//! `hexo-raster = { path = "../hexo-raster" }` to hexo-mcts dependencies.

use hexo_engine::{GameConfig, GameState};
use hexo_mcts::axis_graph::game_to_axis_graph_raw_opts;
use hexo_raster::{RAY_DIRS, RasterSpec, build_raster};

fn ray_and_distance(delta: (i32, i32)) -> Option<(usize, usize)> {
    for (ray, &(dq, dr)) in RAY_DIRS.iter().enumerate() {
        for distance in 1..=5usize {
            if delta == (dq * distance as i32, dr * distance as i32) {
                return Some((ray, distance));
            }
        }
    }
    None
}

fn assert_position_parity(game: &GameState) {
    let graph = game_to_axis_graph_raw_opts(game, true, false, true);
    let raster = build_raster(game, &RasterSpec::default()).unwrap();
    let real_nodes = graph.num_nodes - 1; // exclude dummy
    let mut graph_real_axis_edges = 0usize;

    for edge in 0..graph.edge_src.len() {
        let src = graph.edge_src[edge] as usize;
        let dst = graph.edge_dst[edge] as usize;
        if src >= real_nodes || dst >= real_nodes {
            continue;
        }
        let attr = &graph.edge_attr[edge * 5..edge * 5 + 5];
        if attr[0].abs() + attr[1].abs() + attr[2].abs() <= 0.5 {
            continue;
        }
        let src_coord = (graph.coords[src * 2], graph.coords[src * 2 + 1]);
        let dst_coord = (graph.coords[dst * 2], graph.coords[dst * 2 + 1]);
        let delta = (src_coord.0 - dst_coord.0, src_coord.1 - dst_coord.1);
        let (ray, distance) = ray_and_distance(delta)
            .unwrap_or_else(|| panic!("non-ray graph edge {src_coord:?}->{dst_coord:?}"));
        let dst_cell = raster.coord_to_index(dst_coord).unwrap();
        assert!(
            raster.has_ray_source(dst_cell, ray, distance),
            "missing raster bit for graph edge {src_coord:?}->{dst_coord:?}"
        );
        graph_real_axis_edges += 1;
    }

    let raster_edges: usize = raster
        .ray_bits
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum();
    assert_eq!(
        raster_edges, graph_real_axis_edges,
        "raster contains a line edge absent from the current graph"
    );
}

fn walk_config(config: GameConfig, placements: usize) {
    let mut game = GameState::with_config(config);
    for step in 0..placements {
        if game.is_terminal() {
            break;
        }
        assert_position_parity(&game);
        let legal = game.legal_moves();
        let index = (step * 37 + 11) % legal.len();
        game.apply_move(legal[index]).unwrap();
    }
}

#[test]
fn dense_ray_bits_match_axis_graph_over_curriculum_positions() {
    for (win_length, placement_radius, max_moves, placements) in [
        (4, 2, 30, 20),
        (5, 2, 40, 24),
        (6, 2, 80, 32),
        (6, 4, 120, 40),
        (6, 8, 200, 48),
    ] {
        walk_config(
            GameConfig {
                win_length,
                placement_radius,
                max_moves,
            },
            placements,
        );
    }
}
