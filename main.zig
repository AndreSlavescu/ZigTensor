const std = @import("std");
const Tensor = @import("Tensor/tensor.zig").Tensor;
const ExpressionParser = @import("Tensor/expression_parser.zig").ExpressionParser;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var a = try Tensor.init(
        allocator,
        &[_]f32{1.0, 2.0, 3.0},
        &[_]usize{3},
        .Float
    );
    defer a.deinit();

    var b = try Tensor.init(
        allocator,
        &[_]f32{4.0, 5.0, 6.0},
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

    std.debug.print("Result of expression 'a + b * 2':\n", .{});
    for (result.data) |val| {
        std.debug.print("{d}\n", .{val});
    }
}