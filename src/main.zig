const std = @import("std");
const GPTNeoX = @import("./GPTNeoX.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const args = try std.process.argsAlloc(gpa.allocator());
    defer std.process.argsFree(gpa.allocator(), args);

    var model = try GPTNeoX.load(gpa.allocator(), args[1], 100);
    defer model.deinit();

    const output = try model.eval(gpa.allocator(), &.{}, .{});
    defer gpa.allocator().free(output);
}

const END_KEY = "### End";

pub fn generatePrompt(writer: anytype, instruction: []const u8) !void {
    try writer.print(
        \\Below is an instruction that describes a task. Write a response that appropriately completes the request.
        \\
        \\### Instruction:
        \\{}
        \\
        \\### Response:
        \\
    , .{instruction});
}
