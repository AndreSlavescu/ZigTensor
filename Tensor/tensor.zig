const std = @import("std");

pub const TensorType = enum {
    Float,
    Int,
};

pub const TensorError = error{
    ShapeMismatch,
    DimensionMismatch,
    InvalidOperation,
};

pub const Tensor = struct {
    data: []f32,
    shape: []usize,
    dtype: TensorType,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, data: []const f32, shape: []const usize, dtype: TensorType) !Tensor {
        const data_copy = try allocator.dupe(f32, data);
        const shape_copy = try allocator.dupe(usize, shape);

        return Tensor{
            .data = data_copy,
            .shape = shape_copy,
            .dtype = dtype,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
    }

    pub fn add(self: Tensor, other: Tensor) !Tensor {
        if (self.data.len != other.data.len) {
            return error.ShapeMismatch;
        }

        var result_data = try self.allocator.alloc(f32, self.data.len);
        errdefer self.allocator.free(result_data);

        for (self.data, 0..) |val, i| {
            result_data[i] = val + other.data[i];
        }

        const result_shape = try self.allocator.dupe(usize, self.shape);
        errdefer self.allocator.free(result_shape);

        return Tensor{
            .data = result_data,
            .shape = result_shape,
            .dtype = self.dtype,
            .allocator = self.allocator,
        };
    }

    pub fn multiply(self: Tensor, other: Tensor) !Tensor {
        const is_self_scalar = self.data.len == 1;
        const is_other_scalar = other.data.len == 1;

        if (is_self_scalar or is_other_scalar) {
            const scalar = if (is_self_scalar) self.data[0] else other.data[0];
            const vec = if (is_self_scalar) other else self;
            
            var result_data = try self.allocator.alloc(f32, vec.data.len);
            errdefer self.allocator.free(result_data);

            for (vec.data, 0..) |val, i| {
                result_data[i] = val * scalar;
            }

            const result_shape = try self.allocator.dupe(usize, vec.shape);
            errdefer self.allocator.free(result_shape);

            return Tensor{
                .data = result_data,
                .shape = result_shape,
                .dtype = self.dtype,
                .allocator = self.allocator,
            };
        }

        if (self.data.len != other.data.len) {
            return error.ShapeMismatch;
        }

        var result_data = try self.allocator.alloc(f32, self.data.len);
        errdefer self.allocator.free(result_data);

        for (self.data, 0..) |val, i| {
            result_data[i] = val * other.data[i];
        }

        const result_shape = try self.allocator.dupe(usize, self.shape);
        errdefer self.allocator.free(result_shape);

        return Tensor{
            .data = result_data,
            .shape = result_shape,
            .dtype = self.dtype,
            .allocator = self.allocator,
        };
    }

    pub fn tensor_product(self: Tensor, other: Tensor) !Tensor {
        const result_len = self.data.len * other.data.len;
        var result_data = try self.allocator.alloc(f32, result_len);
        errdefer self.allocator.free(result_data);

        var i: usize = 0;
        for (self.data) |s_val| {
            for (other.data) |o_val| {
                result_data[i] = s_val * o_val;
                i += 1;
            }
        }

        var result_shape = try self.allocator.alloc(usize, self.shape.len + other.shape.len);
        errdefer self.allocator.free(result_shape);
        
        @memcpy(result_shape[0..self.shape.len], self.shape);
        @memcpy(result_shape[self.shape.len..], other.shape);

        return Tensor{
            .data = result_data,
            .shape = result_shape,
            .dtype = self.dtype,
            .allocator = self.allocator,
        };
    }
};