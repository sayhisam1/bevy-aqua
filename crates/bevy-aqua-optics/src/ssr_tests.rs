const SSR: &str = include_str!("ssr.wgsl");
const OPTICS: &str = include_str!("optics.wgsl");

fn constant(name: &str) -> f32 {
    let marker = format!("const {name}: ");
    let tail = SSR.split(&marker).nth(1).unwrap_or_else(|| panic!("{name} missing"));
    let value = tail.split(" = ").nth(1).unwrap().split(';').next().unwrap();
    value.trim_end_matches('u').parse().unwrap()
}

fn body<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let tail = &source[start..];
    &tail[..tail.find("\n}\n").unwrap()]
}

#[test]
fn bisection_lands_every_crossing_inside_the_thickness_band() {
    // Quadratic spacing makes the final linear interval the widest.
    let steps = constant("SSR_LINEAR_STEPS");
    let start = constant("SSR_START");
    let length = constant("SSR_MAX_DISTANCE") - start;
    let last = length * (1.0 - ((steps - 1.0) / steps).powi(2));
    let refined = last / 2.0_f32.powf(constant("SSR_BISECTION_STEPS"));
    assert!(refined < constant("SSR_THICKNESS"), "{refined} m >= thickness");
    assert!(start < constant("SSR_THICKNESS") + refined);
}

#[test]
fn march_loops_stay_single_level() {
    for name in ["screen_space_reflection", "ssr_refine", "ssr_probe", "ssr_resolve"] {
        assert!(body(SSR, name).matches("for (").count() <= 1, "{name} nests loops");
    }
}

#[test]
fn far_tier_never_marches_depth() {
    assert!(!body(OPTICS, "far_field_water").contains("screen_space"));
}
