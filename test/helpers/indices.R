if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}
if (!requireNamespace("yaml", quietly = TRUE)) {
  install.packages("yaml")
}
if (!requireNamespace("tuneR", quietly = TRUE)) {
  install.packages("tuneR")
}
if (!requireNamespace("seewave", quietly = TRUE)) {
  install.packages("seewave", dependencies = TRUE)
}
if (!requireNamespace("soundecology", quietly = TRUE)) {
    install.packages("soundecology", dependencies = TRUE)
}

# load packages
suppressPackageStartupMessages({
  library(seewave)
  library(tuneR)
  library(arrow)
  library(yaml)
  library(soundecology)
})

# fetch args
args <- commandArgs(trailingOnly = TRUE)

audio_path <- args[1]
params_path <- args[2]
output_path <- args[3]
files <- list.files(audio_path, pattern = "\\.wav$", full.names = TRUE)

# load audio parameters
params <- yaml::read_yaml(params_path)

# seewave uses "hanning" instead of "hann" as the params$window argument
window_mapping <- function(x) {
  mapping <- c(
     "hann" = "hanning"
  )
  ifelse(x %in% names(mapping), mapping[x], x)
}

# compute AI's using seewave
print("Extracting features...")

wavs = lapply(files, function(file_path) {
    wav = readWave(file_path)
    # a warning is printed when the sample rate is the same, but we don't care
    wav = suppressWarnings(downsample(wav, params$sr))
})

bis = sapply(wavs, function(wav) {
    # annoyingly the soundecology package dumps to the console
    invisible(capture.output({
      bi <<- bioacoustic_index(wav, fft_w = params$n_fft, min_freq = params$bi_flim[[1]], max_freq = params$bi_flim[[2]])
    }))
    bi$left_area
})

aeis = sapply(wavs, function(wav) {
    # annoyingly the soundecology package dumps to the console
    invisible(capture.output({
      aei <<- acoustic_evenness(wav, max_freq = params$aei_flim[[2]], freq_step = params$bin_step, db_threshold = params$db_threshold)
    }))
    aei$aei_left
})

rmss = sapply(wavs, function(wav) {
    x = wav@left / (2^(wav@bit - 1))
    rs <- sapply(seq(1, length(x), by = params$n_fft), function(i) {
      rms(x[i:min(i+params$n_fft-1, length(x))])
    })
    mean(rs)
})

shs = sapply(wavs, function(wav) {
    spec <- meanspec(wav, params$sr, wl = params$n_fft, wn = window_mapping(params$window), ovlp = ((params$n_fft - params$hop_length) / params$n_fft) * 100, norm=FALSE, plot = FALSE, PSD = FALSE)
    sh(spec)
})

ths = sapply(wavs, function(wav) {
    th <- th(env(wav, params$sr, wl = params$frame_length, wn = window_mapping(params$window), ovlp = 0, norm=FALSE, plot = FALSE))
})

scs = sapply(wavs, function(wav) {
    S <- spectro(wav, params$sr, channel = 1, wl = params$n_fft, ovlp = ((params$n_fft - params$hop_length) / params$n_fft) * 100, wn = window_mapping(params$window), dB = NULL, plot = FALSE, norm = FALSE)
    # normalise each column, treat as a distribution over frequeny bins
    S_norm <- apply(S$amp, 2, function(x) x / sum(x))
    # extract FFT bins, disclude nyquist
    freqs <- head(seq(0, params$sr/2, length.out = (params$n_fft/2) + 1), -1)
    # compute centroids as per: centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])
    centroids = colSums(freqs * S_norm)
    mean(centroids)
})

acis = sapply(wavs, function(wav) {
    ACI(wav, params$sr, channel = 1, wl = params$n_fft, ovlp = ((params$n_fft - params$hop_length) / params$n_fft) * 100, wn = window_mapping(params$window), nbwindows = 1)
})

sfs = sapply(wavs, function(wav) {
    mean(specflux(wav, params$sr, channel = 1, wl = params$n_fft, ovlp = ((params$n_fft - params$hop_length) / params$n_fft) * 100, wn = window_mapping(params$window), norm = TRUE, p = 2, plot = FALSE)[,2])
})

zcrs = sapply(wavs, function(wav) {
    mean(zcr(wav, channel = 1, params$sr, wl = params$n_fft, ovlp = ((params$n_fft - params$hop_length) / params$n_fft) * 100, plot = FALSE)[,2])
})

df <- data.frame(
   file_name = basename(files),
   spectral_centroid = scs,
   acoustic_complexity_index = acis,
   zero_crossing_rate = zcrs,
   spectral_flux = sfs,
   root_mean_square = rmss,
   temporal_entropy = ths,
   spectral_entropy = shs,
   bioacoustic_index = bis,
   acoustic_evenness_index = aeis,
   stringsAsFactors = FALSE
)

# persist dataframe
write_parquet(df, output_path)
