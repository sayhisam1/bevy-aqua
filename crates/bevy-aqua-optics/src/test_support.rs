//! Shared source helpers for shader contract tests.

pub(super) fn wgsl_function<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("fn {name}(");
    let start = source
        .find(&marker)
        .unwrap_or_else(|| panic!("WGSL function missing: {name}"));
    let body_start = start
        + source[start..]
            .find('{')
            .unwrap_or_else(|| panic!("WGSL function body missing: {name}"));
    let mut depth = 0u32;
    for (offset, character) in source[body_start..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' if depth == 1 => return &source[start..=body_start + offset],
            '}' => depth -= 1,
            _ => {}
        }
    }
    panic!("WGSL function body is unclosed: {name}");
}

pub(super) fn compact_wgsl(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split("//").next().unwrap_or_default())
        .flat_map(str::chars)
        .filter(|character| !character.is_whitespace())
        .collect()
}

pub(super) fn compact_wgsl_function(source: &str, name: &str) -> String {
    compact_wgsl(wgsl_function(source, name))
}
