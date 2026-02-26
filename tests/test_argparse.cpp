#include <utils2/test.hpp>
#include <utils2/argparse.hpp>

using namespace utils2;

// Helper to build argv from initializer list
struct Args {
    std::vector<std::string> storage;
    std::vector<const char*> ptrs;

    Args(std::initializer_list<std::string_view> args) {
        storage.reserve(args.size());
        ptrs.reserve(args.size());
        for (auto sv : args) {
            storage.emplace_back(sv);
        }
        for (const auto& s : storage) {
            ptrs.push_back(s.c_str());
        }
    }

    [[nodiscard]] int argc() const noexcept {
        return static_cast<int>(ptrs.size());
    }

    [[nodiscard]] const char* const* argv() const noexcept {
        return ptrs.data();
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_CASE("ArgParser: --key value long option") {
    ArgParser p("test");
    p.option("output", "o", "output file");
    Args a{"test", "--output", "foo.txt"};
    p.parse(a.argc(), a.argv());
    REQUIRE(p.has("output"));
    REQUIRE_EQ(p.get("output"), std::string_view("foo.txt"));
}

TEST_CASE("ArgParser: --key=value long option") {
    ArgParser p("test");
    p.option("output", "o", "output file");
    Args a{"test", "--output=bar.txt"};
    p.parse(a.argc(), a.argv());
    REQUIRE(p.has("output"));
    REQUIRE_EQ(p.get("output"), std::string_view("bar.txt"));
}

TEST_CASE("ArgParser: short option -k value") {
    ArgParser p("test");
    p.option("output", "o", "output file");
    Args a{"test", "-o", "baz.txt"};
    p.parse(a.argc(), a.argv());
    REQUIRE(p.has("output"));
    REQUIRE_EQ(p.get("output"), std::string_view("baz.txt"));
}

TEST_CASE("ArgParser: short option -kvalue (no space)") {
    ArgParser p("test");
    p.option("output", "o", "output file");
    Args a{"test", "-ofile.txt"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get("output"), std::string_view("file.txt"));
}

TEST_CASE("ArgParser: flag present") {
    ArgParser p("test");
    p.flag("verbose", "v", "enable verbose mode");
    Args a{"test", "--verbose"};
    p.parse(a.argc(), a.argv());
    REQUIRE(p.has("verbose"));
    REQUIRE_EQ(p.get("verbose"), std::string_view("true"));
}

TEST_CASE("ArgParser: flag absent defaults to false") {
    ArgParser p("test");
    p.flag("verbose", "v", "enable verbose mode");
    Args a{"test"};
    p.parse(a.argc(), a.argv());
    REQUIRE(p.has("verbose"));
    REQUIRE_EQ(p.get("verbose"), std::string_view("false"));
}

TEST_CASE("ArgParser: short flag") {
    ArgParser p("test");
    p.flag("verbose", "v", "enable verbose mode");
    Args a{"test", "-v"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get("verbose"), std::string_view("true"));
}

TEST_CASE("ArgParser: positional arguments") {
    ArgParser p("test");
    p.positional("input", "input file");
    p.positional("output", "output file");
    Args a{"test", "in.txt", "out.txt"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get("input"), std::string_view("in.txt"));
    REQUIRE_EQ(p.get("output"), std::string_view("out.txt"));

    auto pos = p.positionals();
    REQUIRE_EQ(pos.size(), 2u);
    REQUIRE_EQ(pos[0], std::string("in.txt"));
    REQUIRE_EQ(pos[1], std::string("out.txt"));
}

TEST_CASE("ArgParser: required option throws on missing") {
    ArgParser p("test");
    p.option("output", "o", "output file");
    p.required("output");
    Args a{"test"};
    REQUIRE_THROWS(p.parse(a.argc(), a.argv()));
}

TEST_CASE("ArgParser: required option present does not throw") {
    ArgParser p("test");
    p.option("output", "o", "output file");
    p.required("output");
    Args a{"test", "--output", "ok.txt"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get("output"), std::string_view("ok.txt"));
}

TEST_CASE("ArgParser: unknown option throws") {
    ArgParser p("test");
    Args a{"test", "--bogus"};
    REQUIRE_THROWS(p.parse(a.argc(), a.argv()));
}

TEST_CASE("ArgParser: unknown short option throws") {
    ArgParser p("test");
    Args a{"test", "-z"};
    REQUIRE_THROWS(p.parse(a.argc(), a.argv()));
}

TEST_CASE("ArgParser: default value") {
    ArgParser p("test");
    p.option("count", "c", "item count", "42");
    Args a{"test"};
    p.parse(a.argc(), a.argv());
    REQUIRE(p.has("count"));
    REQUIRE_EQ(p.get("count"), std::string_view("42"));
}

TEST_CASE("ArgParser: default value overridden") {
    ArgParser p("test");
    p.option("count", "c", "item count", "42");
    Args a{"test", "--count", "100"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get("count"), std::string_view("100"));
}

TEST_CASE("ArgParser: get_as<int>") {
    ArgParser p("test");
    p.option("count", "c", "count", "10");
    Args a{"test", "--count", "99"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get_as<int>("count"), 99);
}

TEST_CASE("ArgParser: get_as<int> default") {
    ArgParser p("test");
    p.option("count", "c", "count", "10");
    Args a{"test"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get_as<int>("count"), 10);
}

TEST_CASE("ArgParser: get_as<double>") {
    ArgParser p("test");
    p.option("ratio", "r", "ratio");
    Args a{"test", "--ratio", "3.14"};
    p.parse(a.argc(), a.argv());
    REQUIRE_NEAR(p.get_as<double>("ratio"), 3.14, 1e-9);
}

TEST_CASE("ArgParser: get_as<int> with fallback for missing key") {
    ArgParser p("test");
    Args a{"test"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get_as<int>("missing", 77), 77);
}

TEST_CASE("ArgParser: get with fallback for missing key") {
    ArgParser p("test");
    Args a{"test"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get("missing", "fallback"), std::string_view("fallback"));
}

TEST_CASE("ArgParser: help() text generation") {
    ArgParser p("myapp", "A cool app");
    p.option("output", "o", "output file");
    p.flag("verbose", "v", "enable verbose");
    p.positional("input", "input file");

    auto h = p.help();
    // Should contain program name, description, options, and positional
    REQUIRE(h.find("myapp") != std::string::npos);
    REQUIRE(h.find("A cool app") != std::string::npos);
    REQUIRE(h.find("--output") != std::string::npos);
    REQUIRE(h.find("-o") != std::string::npos);
    REQUIRE(h.find("--verbose") != std::string::npos);
    REQUIRE(h.find("input") != std::string::npos);
    REQUIRE(h.find("--help") != std::string::npos);
}

TEST_CASE("ArgParser: handle_help() detects --help") {
    ArgParser p("test");
    Args a{"test", "--help"};
    REQUIRE(p.handle_help(a.argc(), a.argv()));
}

TEST_CASE("ArgParser: handle_help() detects -h") {
    ArgParser p("test");
    Args a{"test", "-h"};
    REQUIRE(p.handle_help(a.argc(), a.argv()));
}

TEST_CASE("ArgParser: handle_help() returns false when absent") {
    ArgParser p("test");
    Args a{"test", "--verbose"};
    REQUIRE(!p.handle_help(a.argc(), a.argv()));
}

TEST_CASE("ArgParser: -- stops option parsing") {
    ArgParser p("test");
    p.option("output", "o", "output file");
    p.positional("arg1", "first arg");
    Args a{"test", "--", "--output"};
    p.parse(a.argc(), a.argv());
    // "--output" after -- should be treated as positional, not as option
    REQUIRE(!p.has("output"));
    REQUIRE(p.has("arg1"));
    REQUIRE_EQ(p.get("arg1"), std::string_view("--output"));
}

TEST_CASE("ArgParser: required positional throws when missing") {
    ArgParser p("test");
    p.positional("input", "input file", true);
    Args a{"test"};
    REQUIRE_THROWS(p.parse(a.argc(), a.argv()));
}

TEST_CASE("ArgParser: optional positional does not throw") {
    ArgParser p("test");
    p.positional("input", "input file", false);
    Args a{"test"};
    p.parse(a.argc(), a.argv());
    REQUIRE(!p.has("input"));
}

TEST_CASE("ArgParser: get_as<bool> conversion") {
    ArgParser p("test");
    p.flag("verbose", "v", "verbose");
    Args a{"test", "-v"};
    p.parse(a.argc(), a.argv());
    REQUIRE_EQ(p.get_as<bool>("verbose"), true);
}

UTILS2_TEST_MAIN()
