const std = @import("std");
const Tensor = @import("../tensor.zig").Tensor;
const ExpressionParser = @import("../expression_parser.zig").ExpressionParser;

test "basic expression evaluation" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(
        allocator,
        &[_]f32{ 1.0, 2.0, 3.0 },
        &[_]usize{3},
        .Float
    );
    defer a.deinit();

    var b = try Tensor.init(
        allocator,
        &[_]f32{ 4.0, 5.0, 6.0 },
        &[_]usize{3},
        .Float
    );
    defer b.deinit();

    var variables = std.StringHashMap(Tensor).init(allocator);
    defer variables.deinit();
    try variables.put("a", a);
    try variables.put("b", b);

    var result = try ExpressionParser.eval("a + b", variables, allocator);
    defer result.deinit();

    try std.testing.expectEqual(result.data.len, 3);
    try std.testing.expectEqual(result.data[0], 5.0);
    try std.testing.expectEqual(result.data[1], 7.0);
    try std.testing.expectEqual(result.data[2], 9.0);
    std.debug.print("basic expression evaluation passed\n", .{});
}

test "scalar multiplication" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(
        allocator,
        &[_]f32{ 1.0, 2.0, 3.0 },
        &[_]usize{3},
        .Float
    );
    defer a.deinit();

    var two = try Tensor.init(
        allocator,
        &[_]f32{2.0},
        &[_]usize{1},
        .Float
    );
    defer two.deinit();

    var variables = std.StringHashMap(Tensor).init(allocator);
    defer variables.deinit();
    try variables.put("a", a);
    try variables.put("2", two);

    var result = try ExpressionParser.eval("a * 2", variables, allocator);
    defer result.deinit();

    try std.testing.expectEqual(result.data.len, 3);
    try std.testing.expectEqual(result.data[0], 2.0);
    try std.testing.expectEqual(result.data[1], 4.0);
    try std.testing.expectEqual(result.data[2], 6.0);
    std.debug.print("scalar multiplication passed\n", .{});
}

test "complex expression" {
    const allocator = std.testing.allocator;

    var a = try Tensor.init(
        allocator,
        &[_]f32{ 1.0, 2.0, 3.0 },
        &[_]usize{3},
        .Float
    );
    defer a.deinit();

    var b = try Tensor.init(
        allocator,
        &[_]f32{ 4.0, 5.0, 6.0 },
        &[_]usize{3},
        .Float
    );
    defer b.deinit();

    var two = try Tensor.init(
        allocator,
        &[_]f32{2.0},
        &[_]usize{1},
        .Float
    );
    defer two.deinit();

    var variables = std.StringHashMap(Tensor).init(allocator);
    defer variables.deinit();
    try variables.put("a", a);
    try variables.put("b", b);
    try variables.put("2", two);

    var result = try ExpressionParser.eval("a + b * 2", variables, allocator);
    defer result.deinit();

    try std.testing.expectEqual(result.data.len, 3);
    try std.testing.expectEqual(result.data[0], 9.0);  // 1 + (4 * 2)
    try std.testing.expectEqual(result.data[1], 12.0); // 2 + (5 * 2)
    try std.testing.expectEqual(result.data[2], 15.0); // 3 + (6 * 2)
    std.debug.print("complex expression passed\n", .{});
} 
