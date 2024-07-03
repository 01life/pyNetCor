#ifndef OPTIONS_H
#define OPTIONS_H

#include <algorithm>
#include <iostream>
#include <string>

// Supported pearson, spearman and kendall correlation methods
enum class CorrelationMethod {
    Pearson = 1,
    Spearman = 2,
    Kendall = 3
};

inline CorrelationMethod stringToCorrelationMethod(const std::string& str) {
    if (str == "pearson") {
        return CorrelationMethod::Pearson;
    } else if (str == "spearman") {
        return CorrelationMethod::Spearman;
    } else if (str == "kendall") {
        return CorrelationMethod::Kendall;
    } else {
        throw std::runtime_error("Unknown correlation method.");
    }
}

inline std::string toString(const CorrelationMethod corMethod) {
    switch (corMethod) {
        case CorrelationMethod::Pearson:
            return "Pearson";
        case CorrelationMethod::Spearman:
            return "Spearman";
        case CorrelationMethod::Kendall:
            return "Kendall";
        default:
            throw std::runtime_error("Unknown correlation method.");
    }
}

// Supported NAN methods: ignore, fillMean, fillMedian
enum class NAMethod {
    Ignore = 1,
    FillMean = 2,
    FillMedian = 3
};

inline NAMethod stringToNAMethod(const std::string& str) {
    if (str == "ignore") {
        return NAMethod::Ignore;
    } else if (str == "fillMean") {
        return NAMethod::FillMean;
    } else if (str == "fillMedian") {
        return NAMethod::FillMedian;
    } else {
        throw std::runtime_error("Unknown NAN method.");
    }
}

inline std::string toString(const NAMethod naMethod) {
    switch (naMethod) {
        case NAMethod::Ignore:
            return "ignore";
        case NAMethod::FillMean:
            return "fillMean";
        case NAMethod::FillMedian:
            return "fillMedian";
        default:
            throw std::runtime_error("Unknown NAN method.");
    }
}

// Supported distribution types: normal, t
enum class DistributionType {
    Normal = 1,
    T = 2,
};

inline DistributionType stringToDistributionType(const std::string& str) {
    if (str == "normal") {
        return DistributionType::Normal;
    } else if (str == "t") {
        return DistributionType::T;
    } else {
        throw std::runtime_error("Unknown distribution type.");
    }
}

inline std::string toString(const DistributionType distType) {
    switch (distType) {
        case DistributionType::Normal:
            return "normal";
        case DistributionType::T:
            return "t";
        default:
            throw std::runtime_error("Unknown distribution type.");
    }
}

// Supported p-value adjustment methods: holm, hochberg, hommel, bonferroni, BH, BY
enum class PAdjustMethod {
    Holm = 1,
    Hochberg = 2,
    Bonferroni = 3,
    BH = 4,
    BY = 5
};

inline PAdjustMethod stringToPAdjustMethod(const std::string& str) {
    if (str == "holm") {
        return PAdjustMethod::Holm;
    } else if (str == "hochberg") {
        return PAdjustMethod::Hochberg;
    } else if (str == "bonferroni") {
        return PAdjustMethod::Bonferroni;
    } else if (str == "BH") {
        return PAdjustMethod::BH;
    } else if (str == "BY") {
        return PAdjustMethod::BY;
    } else {
        throw std::runtime_error("Unknown p-adjust method.");
    }
}

inline std::string toString(const PAdjustMethod pAdjustMethod) {
    switch (pAdjustMethod) {
        case PAdjustMethod::Holm:
            return "holm";
        case PAdjustMethod::Hochberg:
            return "hochberg";
        case PAdjustMethod::Bonferroni:
            return "bonferroni";
        case PAdjustMethod::BH:
            return "BH";
        case PAdjustMethod::BY:
            return "BY";
        default:
            throw std::runtime_error("Unknown p-adjust method.");
    }
}

// Supported pre-test methods: p, adjust-p
enum class PreTestMethod {
    P = 1,
    PAdjust = 2
};

inline PreTestMethod stringToPreTestMethod(const std::string& str) {
    if (str == "p") {
        return PreTestMethod::P;
    } else if (str == "adjust-p") {
        return PreTestMethod::PAdjust;
    } else {
        throw std::runtime_error("Unknown pre-test method.");
    }
}

inline std::string toString(const PreTestMethod pretestMethod) {
    switch (pretestMethod) {
        case PreTestMethod::P:
            return "p";
        case PreTestMethod::PAdjust:
            return "adjust-p";
        default:
            throw std::runtime_error("Unknown pre-test method.");
    }
}

// Supported distance methods: pearson, spearman
enum class DistanceMethodType {
    Pearson = 1,
    Spearman = 2,
};

inline DistanceMethodType stringToDistanceMethodType(const std::string& str) {
    if (str == "pearson") {
        return DistanceMethodType::Pearson;
    } else if (str == "spearman") {
        return DistanceMethodType::Spearman;
    } else {
        throw std::runtime_error("Unknown distance method.");
    }
}

inline std::string toString(const DistanceMethodType distanceMethodType) {
    switch (distanceMethodType) {
        case DistanceMethodType::Pearson:
            return "pearson";
        case DistanceMethodType::Spearman:
            return "spearman";
        default:
            throw std::runtime_error("Unknown distance method.");
    }
}

// Supported profile measure methods: median, mean, 75Q, 80Q, 85Q, 90Q, 95Q
enum class ProfileMethodType {
    Median = 1,
    Mean = 2,
    Percentile_75 = 3,
    Percentile_80 = 4,
    Percentile_85 = 5,
    Percentile_90 = 6,
    Percentile_95 = 7,
};

inline ProfileMethodType stringToProfileMethodType(const std::string& str) {
    if (str == "median") {
        return ProfileMethodType::Median;
    } else if (str == "mean") {
        return ProfileMethodType::Mean;
    } else if (str == "75Q") {
        return ProfileMethodType::Percentile_75;
    } else if (str == "80Q") {
        return ProfileMethodType::Percentile_80;
    } else if (str == "85Q") {
        return ProfileMethodType::Percentile_85;
    } else if (str == "90Q") {
        return ProfileMethodType::Percentile_90;
    } else if (str == "95Q") {
        return ProfileMethodType::Percentile_95;
    } else {
        throw std::runtime_error("Unknown profile method.");
    }
}

inline std::string toString(const ProfileMethodType profileMethodType) {
    switch (profileMethodType) {
        case ProfileMethodType::Median:
            return "median";
        case ProfileMethodType::Mean:
            return "mean";
        case ProfileMethodType::Percentile_75:
            return "75Q";
        case ProfileMethodType::Percentile_80:
            return "80Q";
        case ProfileMethodType::Percentile_85:
            return "85Q";
        case ProfileMethodType::Percentile_90:
            return "90Q";
        case ProfileMethodType::Percentile_95:
            return "95Q";
        default:
            throw std::runtime_error("Unknown profile method.");
    }
}

// Convert double to string
inline std::string dtos(double d) {
    std::string s = std::to_string(d);

    s.erase(std::find_if(s.rbegin(), s.rend(), [](char ch) {
                return ch != '0' && ch != '.';
            }).base(),
            s.end());
    return s;
}

enum class TopkCorrelationMode {
    Positive = 1,
    Negative = 2,
    Both = 3,
};

inline TopkCorrelationMode stringToTopkCorrelationMode(const std::string& str) {
    if (str == "positive") {
        return TopkCorrelationMode::Positive;
    } else if (str == "negative") {
        return TopkCorrelationMode::Negative;
    } else if (str == "both") {
        return TopkCorrelationMode::Both;
    } else {
        throw std::runtime_error("Unknown correlation method.");
    }
}

inline std::string toString(const TopkCorrelationMode correlationMethod) {
    switch (correlationMethod) {
        case TopkCorrelationMode::Positive:
            return "positive";
        case TopkCorrelationMode::Negative:
            return "negative";
        case TopkCorrelationMode::Both:
            return "both";
        default:
            throw std::runtime_error("Unknown correlation method.");
    }
}

#endif // OPTIONS_H