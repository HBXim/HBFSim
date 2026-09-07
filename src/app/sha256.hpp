#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace hbfsim::app {

class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        if (size > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
            throw std::runtime_error("input is too large to hash");
        }
        total_bytes_ += static_cast<std::uint64_t>(size);
        while (size != 0) {
            const auto take = std::min(size, buffer_.size() - buffered_);
            std::copy_n(
                data,
                take,
                buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_));
            buffered_ += take;
            data += take;
            size -= take;
            if (buffered_ == buffer_.size()) {
                transform(buffer_.data());
                buffered_ = 0;
            }
        }
    }

    [[nodiscard]] std::string finish_hex() {
        if (finished_) {
            throw std::runtime_error("SHA-256 digest was already finalized");
        }
        finished_ = true;
        if (total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8) {
            throw std::runtime_error(
                "input is too large for SHA-256 length encoding");
        }
        const auto bit_length = total_bytes_ * 8;
        buffer_[buffered_++] = 0x80;
        if (buffered_ > 56) {
            std::fill(
                buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                buffer_.end(),
                0);
            transform(buffer_.data());
            buffered_ = 0;
        }
        std::fill(
            buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
            buffer_.begin() + 56,
            0);
        for (std::size_t index = 0; index < 8; ++index) {
            buffer_[63 - index] = static_cast<std::uint8_t>(
                bit_length >> (index * 8));
        }
        transform(buffer_.data());

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto word : state_) {
            out << std::setw(8) << word;
        }
        return out.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> round_constants_ = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    static std::uint32_t rotate_right(
        std::uint32_t value,
        unsigned amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const auto offset = index * 4;
            words[index] =
                (static_cast<std::uint32_t>(block[offset]) << 24) |
                (static_cast<std::uint32_t>(block[offset + 1]) << 16) |
                (static_cast<std::uint32_t>(block[offset + 2]) << 8) |
                static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto s0 = rotate_right(words[index - 15], 7) ^
                rotate_right(words[index - 15], 18) ^
                (words[index - 15] >> 3);
            const auto s1 = rotate_right(words[index - 2], 17) ^
                rotate_right(words[index - 2], 19) ^
                (words[index - 2] >> 10);
            words[index] =
                words[index - 16] + s0 + words[index - 7] + s1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto s1 = rotate_right(e, 6) ^
                rotate_right(e, 11) ^
                rotate_right(e, 25);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto temp1 =
                h + s1 + choose + round_constants_[index] + words[index];
            const auto s0 = rotate_right(a, 2) ^
                rotate_right(a, 13) ^
                rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffered_ = 0;
    std::uint64_t total_bytes_ = 0;
    bool finished_ = false;
};

inline std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "cannot open file for hashing: " + path.string());
    }
    Sha256 hash;
    std::array<char, 64 * 1024> buffer{};
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = in.gcount();
        if (count > 0) {
            hash.update(
                reinterpret_cast<const std::uint8_t*>(buffer.data()),
                static_cast<std::size_t>(count));
        }
    }
    if (!in.eof()) {
        throw std::runtime_error(
            "failed while hashing file: " + path.string());
    }
    return hash.finish_hex();
}

inline std::string sha256_text(std::string_view value) {
    Sha256 hash;
    hash.update(
        reinterpret_cast<const std::uint8_t*>(value.data()),
        value.size());
    return hash.finish_hex();
}

} // namespace hbfsim::app
