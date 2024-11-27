const std = @import("std");
const Tensor = @import("Tensor/tensor.zig").Tensor;

test "basic tensor operations" {
    const allocator = std.testing.allocator;
    
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]usize{ 2, 2 };
    var tensor = try Tensor.init(allocator, &data, &shape, .Float);
    defer tensor.deinit();
    
    try std.testing.expectEqual(tensor.data.len, 4);
    try std.testing.expectEqual(tensor.shape.len, 2);
    try std.testing.expectEqual(tensor.shape[0], 2);
    try std.testing.expectEqual(tensor.shape[1], 2);
    std.debug.print("basic tensor operations passed\n", .{});
}

test "tensor addition" {
    const allocator = std.testing.allocator;
    
    const data1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const data2 = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]usize{ 2, 2 };
    
    var tensor1 = try Tensor.init(allocator, &data1, &shape, .Float);
    defer tensor1.deinit();
    var tensor2 = try Tensor.init(allocator, &data2, &shape, .Float);
    defer tensor2.deinit();
    
    var result = try tensor1.add(tensor2);
    defer result.deinit();
    
    const expected = [_]f32{ 3.0, 5.0, 7.0, 9.0 };
    for (result.data, 0..) |val, i| {
        try std.testing.expectEqual(val, expected[i]);
    }
    std.debug.print("tensor addition passed\n", .{});
}

test "scalar multiplication" {
    const allocator = std.testing.allocator;
    
    const data1 = [_]f32{ 2.0, 3.0, 4.0 };
    const shape1 = [_]usize{3};
    var tensor1 = try Tensor.init(allocator, &data1, &shape1, .Float);
    defer tensor1.deinit();
    
    const scalar_data = [_]f32{2.0};
    const scalar_shape = [_]usize{1};
    var scalar = try Tensor.init(allocator, &scalar_data, &scalar_shape, .Float);
    defer scalar.deinit();
    
    var result = try tensor1.multiply(scalar);
    defer result.deinit();
    
    const expected = [_]f32{ 4.0, 6.0, 8.0 };
    for (result.data, 0..) |val, i| {
        try std.testing.expectEqual(val, expected[i]);
    }
    std.debug.print("scalar multiplication passed\n", .{});
}

test "tensor product" {
    const allocator = std.testing.allocator;
    
    const data1 = [_]f32{ 1.0, 2.0 };
    const shape1 = [_]usize{2};
    var tensor1 = try Tensor.init(allocator, &data1, &shape1, .Float);
    defer tensor1.deinit();
    
    const data2 = [_]f32{ 3.0, 4.0 };
    const shape2 = [_]usize{2};
    var tensor2 = try Tensor.init(allocator, &data2, &shape2, .Float);
    defer tensor2.deinit();
    
    var result = try tensor1.tensor_product(tensor2);
    defer result.deinit();
    
    const expected = [_]f32{ 3.0, 4.0, 6.0, 8.0 };
    try std.testing.expectEqual(result.data.len, 4);
    for (result.data, 0..) |val, i| {
        try std.testing.expectEqual(val, expected[i]);
    }
    std.debug.print("tensor product passed\n", .{});
}
