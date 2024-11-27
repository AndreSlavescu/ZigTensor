const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

pub const TokenType = enum {
    Number,
    Plus,
    Minus,
    Multiply,
    Divide,
    OpenParen,
    CloseParen,
    Identifier,
    EOF,
};

pub const Token = struct {
    type: TokenType,
    value: []const u8,
};

pub const ExprNode = struct {
    const NodeType = enum {
        BinaryOp,
        Identifier,
        Number,
    };

    type: NodeType,
    value: []const u8,
    left: ?*ExprNode = null,
    right: ?*ExprNode = null,

    pub fn deinit(self: *ExprNode, allocator: std.mem.Allocator) void {
        if (self.left) |left| {
            left.deinit(allocator);
            allocator.destroy(left);
        }
        if (self.right) |right| {
            right.deinit(allocator);
            allocator.destroy(right);
        }
    }
};

pub const ParserError = error{
    SyntaxError,
    UnexpectedToken,
    UnmatchedParenthesis,
    InvalidCharacter,
    OutOfMemory,
    UnsupportedOperator,
    UndefinedVariable,
};

pub const ExpressionParser = struct {
    tokens: []const Token,
    current: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(tokens: []const Token, allocator: std.mem.Allocator) ExpressionParser {
        return ExpressionParser{
            .tokens = tokens,
            .allocator = allocator,
            .current = 0,
        };
    }

    pub fn parse(self: *ExpressionParser) ParserError!*ExprNode {
        return self.parseExpression();
    }

    fn parseExpression(self: *ExpressionParser) ParserError!*ExprNode {
        var left = try self.parseTerm();
        errdefer left.deinit(self.allocator);

        while (self.current < self.tokens.len - 1) {
            const token = self.tokens[self.current];
            
            switch (token.type) {
                .Plus, .Minus => {},
                else => break,
            }

            self.current += 1;
            const right = try self.parseTerm();
            errdefer right.deinit(self.allocator);

            const node = try self.allocator.create(ExprNode);
            errdefer self.allocator.destroy(node);
            
            node.* = .{
                .type = .BinaryOp,
                .value = token.value,
                .left = left,
                .right = right,
            };
            left = node;
        }

        return left;
    }

    fn parseTerm(self: *ExpressionParser) ParserError!*ExprNode {
        var left = try self.parseFactor();
        errdefer left.deinit(self.allocator);

        while (self.current < self.tokens.len - 1) {
            const token = self.tokens[self.current];
            
            switch (token.type) {
                .Multiply, .Divide => {},
                else => break,
            }

            self.current += 1;
            const right = try self.parseFactor();
            errdefer right.deinit(self.allocator);

            const node = try self.allocator.create(ExprNode);
            errdefer self.allocator.destroy(node);
            
            node.* = .{
                .type = .BinaryOp,
                .value = token.value,
                .left = left,
                .right = right,
            };
            left = node;
        }

        return left;
    }

    fn parseFactor(self: *ExpressionParser) ParserError!*ExprNode {
        if (self.current >= self.tokens.len) return error.UnexpectedToken;
        
        const token = self.tokens[self.current];
        self.current += 1;

        const node = try self.allocator.create(ExprNode);
        errdefer self.allocator.destroy(node);

        switch (token.type) {
            .Number => {
                node.* = .{
                    .type = .Number,
                    .value = token.value,
                };
            },
            .Identifier => {
                node.* = .{
                    .type = .Identifier,
                    .value = token.value,
                };
            },
            .OpenParen => {
                var expr = try self.parseExpression();
                if (self.current >= self.tokens.len or 
                    self.tokens[self.current].type != .CloseParen) {
                    expr.deinit(self.allocator);
                    return error.UnmatchedParenthesis;
                }
                self.current += 1;
                return expr;
            },
            else => return error.UnexpectedToken,
        }

        return node;
    }

    pub fn comptime_eval(comptime expr: []const u8, variables: std.StringHashMap(Tensor)) !Tensor {
        const tokens = try tokenize(expr, variables.allocator);
        defer freeTokens(tokens, variables.allocator);

        var parser = ExpressionParser.init(tokens, variables.allocator);
        var ast = try parser.parse();
        defer {
            ast.deinit(variables.allocator);
            variables.allocator.destroy(ast);
        }

        return evalNode(ast, variables);
    }

    pub fn eval(expr: []const u8, variables: std.StringHashMap(Tensor), allocator: std.mem.Allocator) !Tensor {
        const tokens = try tokenize(expr, allocator);
        defer freeTokens(tokens, allocator);

        var parser = ExpressionParser.init(tokens, allocator);
        var ast = try parser.parse();
        defer {
            ast.deinit(allocator);
            allocator.destroy(ast);
        }

        return evalNode(ast, variables);
    }
};

pub fn tokenize(expr: []const u8, allocator: std.mem.Allocator) ![]Token {
    var tokens = std.ArrayList(Token).init(allocator);
    errdefer tokens.deinit();
    
    var i: usize = 0;
    while (i < expr.len) {
        const c = expr[i];
        switch (c) {
            ' ', '\t', '\n', '\r' => i += 1,
            '0'...'9' => {
                const num_start = i;
                while (i < expr.len and (std.ascii.isDigit(expr[i]) or expr[i] == '.')) {
                    i += 1;
                }
                const value = try allocator.dupe(u8, expr[num_start..i]);
                errdefer allocator.free(value);
                try tokens.append(Token{
                    .type = .Number,
                    .value = value,
                });
            },
            'a'...'z', 'A'...'Z', '_' => {
                const id_start = i;
                while (i < expr.len and (std.ascii.isAlphanumeric(expr[i]) or expr[i] == '_')) {
                    i += 1;
                }
                const value = try allocator.dupe(u8, expr[id_start..i]);
                errdefer allocator.free(value);
                try tokens.append(Token{
                    .type = .Identifier,
                    .value = value,
                });
            },
            '+' => {
                const value = try allocator.dupe(u8, expr[i..i+1]);
                try tokens.append(Token{ .type = .Plus, .value = value });
                i += 1;
            },
            '*' => {
                const value = try allocator.dupe(u8, expr[i..i+1]);
                try tokens.append(Token{ .type = .Multiply, .value = value });
                i += 1;
            },
            else => return error.InvalidCharacter,
        }
    }

    const eof_value = try allocator.dupe(u8, "");
    try tokens.append(Token{ .type = .EOF, .value = eof_value });
    
    return tokens.toOwnedSlice();
}

pub fn freeTokens(tokens: []Token, allocator: std.mem.Allocator) void {
    for (tokens) |token| {
        allocator.free(token.value);
    }
    allocator.free(tokens);
}

pub fn evalNode(node: *const ExprNode, variables: std.StringHashMap(Tensor)) !Tensor {
    switch (node.type) {
        .BinaryOp => {
            var left = try evalNode(node.left.?, variables);
            defer left.deinit();
            var right = try evalNode(node.right.?, variables);
            defer right.deinit();
            
            if (std.mem.eql(u8, node.value, "+")) {
                return left.add(right);
            } else if (std.mem.eql(u8, node.value, "*")) {
                return left.multiply(right);
            } else if (std.mem.eql(u8, node.value, "⊗")) {
                return left.tensor_product(right);
            }
            return error.UnsupportedOperator;
        },
        .Identifier => {
            const tensor = variables.get(node.value) orelse return error.UndefinedVariable;
            return Tensor.init(
                tensor.allocator,
                tensor.data,
                tensor.shape,
                tensor.dtype
            );
        },
        .Number => {
            const val = try std.fmt.parseFloat(f32, node.value);
            return Tensor.init(
                variables.allocator,
                &[_]f32{val},
                &[_]usize{1},
                .Float
            );
        },
    }
}
