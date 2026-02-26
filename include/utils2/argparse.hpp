#pragma once
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace utils2 {

// ---------------------------------------------------------------------------
// ArgParser — simple composable command-line argument parser
// ---------------------------------------------------------------------------

class ArgParser final {
public:
    explicit ArgParser(std::string_view program_name,
                       std::string_view description = "")
        : program_name_(program_name), description_(description) {}

    // -- Define options -----------------------------------------------------

    ArgParser& option(std::string_view long_name, std::string_view short_name,
                      std::string_view help, std::string_view default_value = "") {
        opts_.push_back({std::string(long_name), std::string(short_name),
                         std::string(help), std::string(default_value),
                         /*is_flag=*/false, /*required=*/false});
        return *this;
    }

    ArgParser& flag(std::string_view long_name, std::string_view short_name,
                    std::string_view help) {
        opts_.push_back({std::string(long_name), std::string(short_name),
                         std::string(help), "", /*is_flag=*/true, /*required=*/false});
        return *this;
    }

    ArgParser& positional(std::string_view name, std::string_view help,
                          bool required = true) {
        pos_defs_.push_back(
            {std::string(name), std::string(help), required});
        return *this;
    }

    ArgParser& required(std::string_view long_name) {
        for (auto& o : opts_) {
            if (o.long_name == long_name) { o.required = true; return *this; }
        }
        throw std::runtime_error(
            "ArgParser::required: unknown option '--" + std::string(long_name) + "'");
    }

    // -- Parse --------------------------------------------------------------

    void parse(int argc, const char* const* argv) {
        // Reset state
        values_.clear();
        positionals_.clear();

        // Set defaults
        for (const auto& o : opts_) {
            if (o.is_flag) {
                values_.push_back({o.long_name, "false"});
            } else if (!o.default_value.empty()) {
                values_.push_back({o.long_name, o.default_value});
            }
        }

        bool dashdash = false;
        std::size_t pos_idx = 0;

        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);

            if (dashdash) {
                store_positional(pos_idx, std::string(arg));
                continue;
            }

            if (arg == "--") { dashdash = true; continue; }

            // Long option
            if (arg.starts_with("--")) {
                auto body = arg.substr(2);
                auto eq = body.find('=');
                auto key = std::string(eq != std::string_view::npos
                                           ? body.substr(0, eq) : body);
                const auto* opt = find_long(key);
                if (!opt)
                    throw std::runtime_error("Unknown option '--" + key + "'");

                if (opt->is_flag) {
                    set_value(opt->long_name, "true");
                } else if (eq != std::string_view::npos) {
                    set_value(opt->long_name, std::string(body.substr(eq + 1)));
                } else {
                    if (i + 1 >= argc)
                        throw std::runtime_error(
                            "Option '--" + key + "' requires a value");
                    set_value(opt->long_name, std::string(argv[++i]));
                }
                continue;
            }

            // Short option
            if (arg.starts_with("-") && arg.size() > 1) {
                char ch = arg[1];
                const auto* opt = find_short(ch);
                if (!opt)
                    throw std::runtime_error(
                        std::string("Unknown option '-") + ch + "'");

                if (opt->is_flag) {
                    set_value(opt->long_name, "true");
                } else if (arg.size() > 2) {
                    // -kvalue
                    set_value(opt->long_name, std::string(arg.substr(2)));
                } else {
                    if (i + 1 >= argc)
                        throw std::runtime_error(
                            std::string("Option '-") + ch + "' requires a value");
                    set_value(opt->long_name, std::string(argv[++i]));
                }
                continue;
            }

            // Positional
            store_positional(pos_idx, std::string(arg));
        }

        // Check required options
        for (const auto& o : opts_) {
            if (o.required && !has(o.long_name))
                throw std::runtime_error(
                    "Missing required option '--" + o.long_name + "'");
        }

        // Check required positionals
        for (std::size_t j = 0; j < pos_defs_.size(); ++j) {
            if (pos_defs_[j].required && !has(pos_defs_[j].name))
                throw std::runtime_error(
                    "Missing required positional argument '" +
                    pos_defs_[j].name + "'");
        }
    }

    // -- Query results ------------------------------------------------------

    [[nodiscard]] bool has(std::string_view name) const noexcept {
        return find_value(name) != nullptr;
    }

    [[nodiscard]] std::string_view get(std::string_view name) const {
        const auto* v = find_value(name);
        if (!v)
            throw std::runtime_error(
                "No value for '" + std::string(name) + "'");
        return *v;
    }

    [[nodiscard]] std::string_view get(std::string_view name,
                                       std::string_view fallback) const noexcept {
        const auto* v = find_value(name);
        return v ? std::string_view(*v) : fallback;
    }

    // -- Typed getters ------------------------------------------------------

    template <typename T>
    [[nodiscard]] T get_as(std::string_view name) const {
        return convert<T>(get(name), name);
    }

    template <typename T>
    [[nodiscard]] T get_as(std::string_view name, T fallback) const noexcept {
        const auto* v = find_value(name);
        if (!v) return fallback;
        try { return convert<T>(*v, name); }
        catch (...) { return fallback; }
    }

    // -- Positionals --------------------------------------------------------

    [[nodiscard]] std::span<const std::string> positionals() const noexcept {
        return positionals_;
    }

    // -- Help ---------------------------------------------------------------

    [[nodiscard]] std::string help() const {
        std::string out;
        out += "Usage: ";
        out += program_name_;
        if (!opts_.empty()) out += " [OPTIONS]";
        for (const auto& p : pos_defs_) {
            out += p.required ? " <" : " [";
            out += p.name;
            out += p.required ? ">" : "]";
        }
        out += "\n";

        if (!description_.empty()) {
            out += "\n";
            out += description_;
            out += "\n";
        }

        if (!pos_defs_.empty()) {
            out += "\nPositional arguments:\n";
            for (const auto& p : pos_defs_) {
                out += "  ";
                out += p.name;
                pad(out, p.name.size(), 20);
                out += p.help;
                if (!p.required) out += " (optional)";
                out += "\n";
            }
        }

        if (!opts_.empty()) {
            out += "\nOptions:\n";
            for (const auto& o : opts_) {
                std::string col;
                if (!o.short_name.empty()) {
                    col += "-";
                    col += o.short_name;
                    col += ", ";
                } else {
                    col += "    ";
                }
                col += "--";
                col += o.long_name;
                if (!o.is_flag) col += " <value>";

                out += "  ";
                out += col;
                pad(out, col.size(), 28);
                out += o.help;
                if (o.required) out += " (required)";
                else if (!o.is_flag && !o.default_value.empty())
                    out += " [default: " + o.default_value + "]";
                out += "\n";
            }
        }

        out += "  -h, --help                  Show this help message\n";
        return out;
    }

    bool handle_help(int argc, const char* const* argv) const {
        for (int i = 1; i < argc; ++i) {
            std::string_view a(argv[i]);
            if (a == "--help" || a == "-h") {
                std::fputs(help().c_str(), stdout);
                return true;
            }
            if (a == "--") break;
        }
        return false;
    }

private:
    // -- Internal types -----------------------------------------------------

    struct OptDef {
        std::string long_name;
        std::string short_name;   // single char as string
        std::string help;
        std::string default_value;
        bool        is_flag  = false;
        bool        required = false;
    };

    struct PosDef {
        std::string name;
        std::string help;
        bool        required = true;
    };

    struct KV {
        std::string key;
        std::string value;
    };

    // -- Data ---------------------------------------------------------------

    std::string        program_name_;
    std::string        description_;
    std::vector<OptDef> opts_;
    std::vector<PosDef> pos_defs_;
    std::vector<KV>     values_;
    std::vector<std::string> positionals_;

    // -- Helpers ------------------------------------------------------------

    const OptDef* find_long(std::string_view name) const noexcept {
        for (const auto& o : opts_)
            if (o.long_name == name) return &o;
        return nullptr;
    }

    const OptDef* find_short(char ch) const noexcept {
        for (const auto& o : opts_)
            if (!o.short_name.empty() && o.short_name[0] == ch) return &o;
        return nullptr;
    }

    void set_value(const std::string& key, std::string val) {
        for (auto& kv : values_) {
            if (kv.key == key) { kv.value = std::move(val); return; }
        }
        values_.push_back({key, std::move(val)});
    }

    void store_positional(std::size_t& idx, std::string val) {
        if (idx < pos_defs_.size()) {
            set_value(pos_defs_[idx].name, val);
            ++idx;
        }
        positionals_.push_back(std::move(val));
    }

    const std::string* find_value(std::string_view name) const noexcept {
        for (const auto& kv : values_)
            if (kv.key == name) return &kv.value;
        return nullptr;
    }

    static void pad(std::string& s, std::size_t cur, std::size_t target) {
        if (cur < target)
            s.append(target - cur, ' ');
        else
            s += "  ";
    }

    template <typename T>
    static T convert(std::string_view sv, std::string_view name) {
        if constexpr (std::is_same_v<T, std::string>) {
            return std::string(sv);
        } else if constexpr (std::is_same_v<T, std::string_view>) {
            return sv;
        } else if constexpr (std::is_same_v<T, bool>) {
            return sv == "true" || sv == "1" || sv == "yes";
        } else if constexpr (std::is_integral_v<T>) {
            T val{};
            auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
            if (ec != std::errc{})
                throw std::runtime_error(
                    "Cannot convert '" + std::string(sv) +
                    "' to integer for '" + std::string(name) + "'");
            return val;
        } else if constexpr (std::is_floating_point_v<T>) {
            T val{};
            auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
            if (ec != std::errc{})
                throw std::runtime_error(
                    "Cannot convert '" + std::string(sv) +
                    "' to float for '" + std::string(name) + "'");
            return val;
        } else {
            static_assert(!sizeof(T), "Unsupported type for get_as<T>");
        }
    }
};

} // namespace utils2
