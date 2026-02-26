#pragma once

// Only compile if curl is available
#if __has_include(<curl/curl.h>)
#define UTILS2_HAS_CURL 1

#include <curl/curl.h>
#include <string>
#include <string_view>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <functional>
#include <utility>
#include <span>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <cstring>

namespace utils2 {

// ---------------------------------------------------------------------------
// RAII curl handle wrapper (detail)
// ---------------------------------------------------------------------------
namespace detail {

struct CurlDeleter {
    void operator()(CURL* c) const noexcept { curl_easy_cleanup(c); }
};
using CurlHandle = std::unique_ptr<CURL, CurlDeleter>;

// Global init/cleanup (reference counted)
struct CurlGlobal {
    CurlGlobal() {
        if (ref_count_.fetch_add(1, std::memory_order_relaxed) == 0)
            curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    ~CurlGlobal() {
        if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            curl_global_cleanup();
    }
    CurlGlobal(const CurlGlobal&) = delete;
    CurlGlobal& operator=(const CurlGlobal&) = delete;

private:
    static inline std::atomic<int> ref_count_{0};
};

inline CurlHandle make_curl() {
    static CurlGlobal global;
    return CurlHandle{curl_easy_init()};
}

} // namespace detail

// ---------------------------------------------------------------------------
// HttpResponse
// ---------------------------------------------------------------------------
struct HttpResponse {
    long status_code = 0;
    std::vector<std::byte> body;
    std::string content_type;
    std::size_t content_length = 0;

    [[nodiscard]] bool ok() const noexcept { return status_code >= 200 && status_code < 300; }
    [[nodiscard]] bool not_found() const noexcept { return status_code == 404; }

    [[nodiscard]] std::string_view body_string() const noexcept {
        return {reinterpret_cast<const char*>(body.data()), body.size()};
    }
};

// ---------------------------------------------------------------------------
// HttpAuth
// ---------------------------------------------------------------------------
struct HttpAuth {
    std::string bearer_token;
    std::string username;
    std::string password;

    [[nodiscard]] bool empty() const noexcept {
        return bearer_token.empty() && username.empty() && password.empty();
    }
};

// ---------------------------------------------------------------------------
// S3 URL utilities
// ---------------------------------------------------------------------------
struct S3Url {
    std::string bucket;
    std::string key;
    std::string region; // empty if not specified
};

/// Recognises "s3://", "S3://", and "s3+REGION://" (e.g. "s3+us-west-2://").
[[nodiscard]] inline bool is_s3_url(std::string_view url) noexcept {
    if (url.starts_with("s3://") || url.starts_with("S3://"))
        return true;
    // s3+REGION:// form
    if ((url.starts_with("s3+") || url.starts_with("S3+")) && url.find("://") != std::string_view::npos)
        return true;
    return false;
}

/// Parse an S3 URL. Supports "s3://bucket/key" and "s3+REGION://bucket/key".
[[nodiscard]] inline std::optional<S3Url> parse_s3_url(std::string_view url) {
    if (!is_s3_url(url))
        return std::nullopt;

    std::string region;

    // Check for s3+REGION:// form
    auto scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos)
        return std::nullopt;

    auto scheme = url.substr(0, scheme_end);
    auto plus = scheme.find('+');
    if (plus != std::string_view::npos) {
        region = std::string{scheme.substr(plus + 1)};
    }

    auto rest = url.substr(scheme_end + 3); // skip "SCHEME://"
    auto slash = rest.find('/');
    if (slash == std::string_view::npos)
        return S3Url{std::string{rest}, {}, std::move(region)};

    auto bucket = rest.substr(0, slash);
    auto key = rest.substr(slash + 1);
    return S3Url{std::string{bucket}, std::string{key}, std::move(region)};
}

/// Convert an S3 URL to HTTPS. Uses the region from the URL if present,
/// otherwise falls back to path-style without a region subdomain.
[[nodiscard]] inline std::string s3_to_https(const S3Url& parsed) {
    std::string result = "https://";
    result += parsed.bucket;
    if (!parsed.region.empty()) {
        result += ".s3.";
        result += parsed.region;
        result += ".amazonaws.com";
    } else {
        result += ".s3.amazonaws.com";
    }
    if (!parsed.key.empty()) {
        result += '/';
        result += parsed.key;
    }
    return result;
}

/// Convert an S3 URL string to HTTPS.
[[nodiscard]] inline std::string s3_to_https(std::string_view s3_url) {
    auto parsed = parse_s3_url(s3_url);
    if (!parsed)
        return std::string{s3_url};
    return s3_to_https(*parsed);
}

// ---------------------------------------------------------------------------
// AwsAuth -- AWS SigV4 credentials
// ---------------------------------------------------------------------------
struct AwsAuth {
    std::string access_key;
    std::string secret_key;
    std::string session_token; // optional (STS temporary credentials)
    std::string region;        // e.g. "us-east-1"

    [[nodiscard]] bool empty() const noexcept {
        return access_key.empty() && secret_key.empty();
    }

    /// Read credentials from environment variables:
    ///   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    ///   AWS_SESSION_TOKEN, AWS_DEFAULT_REGION
    [[nodiscard]] static AwsAuth from_env() {
        AwsAuth auth;
        if (auto* v = std::getenv("AWS_ACCESS_KEY_ID"))      auth.access_key = v;
        if (auto* v = std::getenv("AWS_SECRET_ACCESS_KEY"))   auth.secret_key = v;
        if (auto* v = std::getenv("AWS_SESSION_TOKEN"))       auth.session_token = v;
        if (auto* v = std::getenv("AWS_DEFAULT_REGION"))      auth.region = v;
        return auth;
    }
};

/// Apply AWS SigV4 authentication to a CURL handle.
/// The returned slist must outlive the curl_easy_perform() call.
/// Pass nullptr for the return value if you don't need it (caller frees).
[[nodiscard]] inline struct curl_slist* apply_aws_auth(CURL* curl, const AwsAuth& auth) {
    if (auth.empty()) return nullptr;

    std::string sigv4 = "aws:amz:" + auth.region + ":s3";
    curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, sigv4.c_str());

    std::string userpwd = auth.access_key + ":" + auth.secret_key;
    curl_easy_setopt(curl, CURLOPT_USERPWD, userpwd.c_str());

    struct curl_slist* headers = nullptr;
    if (!auth.session_token.empty()) {
        std::string hdr = "x-amz-security-token: " + auth.session_token;
        headers = curl_slist_append(headers, hdr.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
    return headers;
}

// ---------------------------------------------------------------------------
// HttpClient
// ---------------------------------------------------------------------------
class HttpClient final {
public:
    struct Config {
        HttpAuth auth{};
        AwsAuth aws_auth{};  // AWS SigV4 authentication (takes precedence over auth if non-empty)
        std::chrono::seconds connect_timeout{10};
        std::chrono::seconds transfer_timeout{30};
        bool follow_redirects{true};
        std::size_t max_retries{3};
        std::string user_agent{"utils2-http/1.0"};
    };

    HttpClient()
        : config_{}
        , handle_(detail::make_curl())
    {
        if (!handle_)
            throw std::runtime_error("http_fetch: failed to create curl handle");
    }

    explicit HttpClient(Config config)
        : config_(std::move(config))
        , handle_(detail::make_curl())
    {
        if (!handle_)
            throw std::runtime_error("http_fetch: failed to create curl handle");
    }

    ~HttpClient() = default;

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    HttpClient(HttpClient&&) noexcept = default;
    HttpClient& operator=(HttpClient&&) noexcept = default;

    // GET request
    [[nodiscard]] HttpResponse get(std::string_view url) const {
        return perform(url, Method::GET, {}, {});
    }

    // GET with byte range
    [[nodiscard]] HttpResponse get_range(std::string_view url,
                                          std::size_t offset, std::size_t length) const {
        auto range = std::to_string(offset) + "-" + std::to_string(offset + length - 1);
        return perform(url, Method::GET_RANGE, {}, {}, range);
    }

    // HEAD request (metadata only)
    [[nodiscard]] HttpResponse head(std::string_view url) const {
        return perform(url, Method::HEAD, {}, {});
    }

    // PUT request
    [[nodiscard]] HttpResponse put(std::string_view url,
                                    std::span<const std::byte> data,
                                    std::string_view content_type = "application/octet-stream") const {
        return perform(url, Method::PUT, data, content_type);
    }

    // Check if URL exists (HEAD + check status)
    [[nodiscard]] bool exists(std::string_view url) const {
        return head(url).ok();
    }

    // Download to file
    bool download(std::string_view url, const std::filesystem::path& dest) const {
        auto resp = get(url);
        if (!resp.ok())
            return false;

        std::ofstream out(dest, std::ios::binary);
        if (!out)
            return false;

        out.write(reinterpret_cast<const char*>(resp.body.data()),
                  static_cast<std::streamsize>(resp.body.size()));
        return out.good();
    }

private:
    enum class Method { GET, GET_RANGE, HEAD, PUT };

    // Write callback: append to std::vector<std::byte>
    static std::size_t write_callback(char* ptr, std::size_t size,
                                       std::size_t nmemb, void* userdata) noexcept {
        auto& buf = *static_cast<std::vector<std::byte>*>(userdata);
        auto total = size * nmemb;
        auto* src = reinterpret_cast<const std::byte*>(ptr);
        buf.insert(buf.end(), src, src + total);
        return total;
    }

    // Header callback: capture content-type and content-length
    static std::size_t header_callback(char* ptr, std::size_t size,
                                        std::size_t nmemb, void* userdata) noexcept {
        auto& resp = *static_cast<HttpResponse*>(userdata);
        auto total = size * nmemb;
        std::string_view line(ptr, total);

        // Strip trailing \r\n
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.remove_suffix(1);

        constexpr std::string_view ct_prefix = "content-type:";
        constexpr std::string_view cl_prefix = "content-length:";

        auto lower_match = [&](std::string_view prefix) -> bool {
            if (line.size() < prefix.size()) return false;
            for (std::size_t i = 0; i < prefix.size(); ++i) {
                char c = line[i];
                if (c >= 'A' && c <= 'Z') c += 32;
                if (c != prefix[i]) return false;
            }
            return true;
        };

        if (lower_match(ct_prefix)) {
            auto val = line.substr(ct_prefix.size());
            while (!val.empty() && val.front() == ' ') val.remove_prefix(1);
            resp.content_type = std::string{val};
        } else if (lower_match(cl_prefix)) {
            auto val = line.substr(cl_prefix.size());
            while (!val.empty() && val.front() == ' ') val.remove_prefix(1);
            std::size_t len = 0;
            for (char c : val) {
                if (c < '0' || c > '9') break;
                len = len * 10 + static_cast<std::size_t>(c - '0');
            }
            resp.content_length = len;
        }

        return total;
    }

    [[nodiscard]] std::string resolve_url(std::string_view url) const {
        if (is_s3_url(url))
            return s3_to_https(url);
        return std::string{url};
    }

    void apply_auth(CURL* curl) const {
        if (!config_.aws_auth.empty()) {
            // AWS SigV4 takes precedence
            aws_sigv4_str_ = "aws:amz:" + config_.aws_auth.region + ":s3";
            curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, aws_sigv4_str_.c_str());
            aws_userpwd_ = config_.aws_auth.access_key + ":" + config_.aws_auth.secret_key;
            curl_easy_setopt(curl, CURLOPT_USERPWD, aws_userpwd_.c_str());
            if (!config_.aws_auth.session_token.empty()) {
                auto hdr = "x-amz-security-token: " + config_.aws_auth.session_token;
                auth_headers_ = nullptr;
                auth_headers_ = curl_slist_append(auth_headers_, hdr.c_str());
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, auth_headers_);
            }
        } else if (!config_.auth.bearer_token.empty()) {
            auto header = "Authorization: Bearer " + config_.auth.bearer_token;
            auth_headers_ = nullptr;
            auth_headers_ = curl_slist_append(auth_headers_, header.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, auth_headers_);
        } else if (!config_.auth.username.empty()) {
            auto userpwd = config_.auth.username + ":" + config_.auth.password;
            curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
            curl_easy_setopt(curl, CURLOPT_USERPWD, userpwd.c_str());
        }
    }

    void cleanup_auth() const noexcept {
        if (auth_headers_) {
            curl_slist_free_all(auth_headers_);
            auth_headers_ = nullptr;
        }
    }

    [[nodiscard]] HttpResponse perform(std::string_view url,
                                        Method method,
                                        std::span<const std::byte> put_data,
                                        std::string_view content_type,
                                        std::string range = {}) const {
        auto resolved = resolve_url(url);
        HttpResponse resp;

        for (std::size_t attempt = 0; attempt <= config_.max_retries; ++attempt) {
            resp = HttpResponse{};
            curl_easy_reset(handle_.get());
            auto* curl = handle_.get();

            // URL
            curl_easy_setopt(curl, CURLOPT_URL, resolved.c_str());

            // Timeouts
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                             static_cast<long>(config_.connect_timeout.count()));
            curl_easy_setopt(curl, CURLOPT_TIMEOUT,
                             static_cast<long>(config_.transfer_timeout.count()));

            // Redirects
            if (config_.follow_redirects) {
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
            }

            // User agent
            curl_easy_setopt(curl, CURLOPT_USERAGENT, config_.user_agent.c_str());

            // Write callback (body)
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);

            // Header callback
            curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header_callback);
            curl_easy_setopt(curl, CURLOPT_HEADERDATA, &resp);

            // Auth
            apply_auth(curl);

            // Method-specific setup
            struct curl_slist* extra_headers = nullptr;

            switch (method) {
                case Method::GET:
                    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
                    break;
                case Method::GET_RANGE:
                    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
                    curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
                    break;
                case Method::HEAD:
                    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
                    break;
                case Method::PUT: {
                    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
                    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE,
                                     static_cast<curl_off_t>(put_data.size()));

                    // Set up read data for PUT
                    put_state_ = PutState{put_data, 0};
                    curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_callback);
                    curl_easy_setopt(curl, CURLOPT_READDATA,
                                     const_cast<PutState*>(&put_state_));

                    if (!content_type.empty()) {
                        auto ct = "Content-Type: " + std::string{content_type};
                        extra_headers = curl_slist_append(extra_headers, ct.c_str());
                        // Merge with auth headers if present
                        if (auth_headers_) {
                            for (auto* node = auth_headers_; node; node = node->next)
                                extra_headers = curl_slist_append(extra_headers, node->data);
                            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, extra_headers);
                            cleanup_auth();
                        } else {
                            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, extra_headers);
                        }
                    }
                    break;
                }
            }

            auto code = curl_easy_perform(curl);

            // Clean up request-scoped slist
            if (extra_headers)
                curl_slist_free_all(extra_headers);
            cleanup_auth();

            if (code == CURLE_OK) {
                curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status_code);

                // Retry on 5xx server errors
                if (resp.status_code >= 500 && attempt < config_.max_retries) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(200 * (1u << attempt)));
                    continue;
                }
                return resp;
            }

            // Retry on network / transient curl errors
            if (attempt < config_.max_retries) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(200 * (1u << attempt)));
                continue;
            }
        }

        return resp; // return last response even on failure
    }

    // Read callback for PUT uploads
    struct PutState {
        std::span<const std::byte> data;
        std::size_t offset = 0;
    };

    static std::size_t read_callback(char* buffer, std::size_t size,
                                      std::size_t nmemb, void* userdata) noexcept {
        auto& state = *static_cast<PutState*>(userdata);
        auto remaining = state.data.size() - state.offset;
        auto to_copy = std::min(size * nmemb, remaining);
        std::memcpy(buffer, state.data.data() + state.offset, to_copy);
        state.offset += to_copy;
        return to_copy;
    }

    Config config_;
    detail::CurlHandle handle_;
    mutable struct curl_slist* auth_headers_ = nullptr;
    mutable PutState put_state_;
    mutable std::string aws_sigv4_str_;  // kept alive for CURLOPT_AWS_SIGV4
    mutable std::string aws_userpwd_;    // kept alive for CURLOPT_USERPWD
};

} // namespace utils2

#endif // __has_include(<curl/curl.h>)
