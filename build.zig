const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zigtensor",
        .root_source_file = .{ .cwd_relative = "main.zig" },
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addAnonymousImport("tensor", .{
        .root_source_file = .{ .cwd_relative = "Tensor/tensor.zig" },
    });

    exe.root_module.addAnonymousImport("expression_parser", .{
        .root_source_file = .{ .cwd_relative = "Tensor/expression_parser.zig" },
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
} 