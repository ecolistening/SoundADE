if (!requireNamespace("seewave", quietly = TRUE)) {
  install.packages("seewave")
}
if (!requireNamespace("tuneR", quietly = TRUE)) {
  install.packages("tuneR")
}
if (!requireNamespace("soundecology", quietly = TRUE)) {
    install.packages("soundecology", dependencies = TRUE)
}
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}
if (!requireNamespace("yaml", quietly = TRUE)) {
  install.packages("yaml")
}
suppressPackageStartupMessages({
  library(seewave)
  library(tuneR)
  library(arrow)
  library(yaml)
  library(soundecology)
})

args <- commandArgs(trailingOnly = TRUE)

audio_path <- args[1]
params_path <- args[2]
output_path <- args[3]
results_path <- file.path(dirname(output_path), "results")

params <- yaml::read_yaml(params_path)

# but seewave uses "hanning" instead of "hann" as the params$window argument
window_mapping <- function(x) {
  mapping <- c(
     "hann" = "hanning"
  )
  ifelse(x %in% names(mapping), mapping[x], x)
}

dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

files <- list.files(audio_path, pattern = "\\.wav$", full.names = TRUE)

# compute using seewave

wavs = lapply(files, function(file_path) {
    wav = readWave(file_path)
    # adjust integer amplitudes to floats
    wav@left / (2^(wav@bit - 1))
})

rmss = sapply(wavs, function(wav) {
    rs <- sapply(seq(1, length(wav), by = params$n_fft), function(i) {
      rms(wav[i:min(i+params$n_fft-1, length(wav))])
    })
    mean(rs)
})
shs = sapply(wavs, function(wav) {
    S <- spectro(wav, params$sr, channel = 1, wl = params$n_fft, ovlp = ((params$n_fft - params$hop_length) / params$n_fft) * 100, wn = window_mapping(params$window), dB = NULL, plot = FALSE)
    S_pow = S$amp ** 2
    sh(S_pow, alpha = "shannon")
})

ths = sapply(wavs, function(wav) {
    max_amp <- sapply(seq(1, length(wav), by = params$n_fft), function(i) {
        max(abs(wav[i:min(i+params$n_fft-1, length(wav))]))
    })
    max_amp <- max_amp / sum(max_amp)
    th(max_amp, breaks = 30)
})

scs = sapply(wavs, function(wav) {
    S <- spectro(wav, params$sr, channel = 1, wl = params$n_fft, ovlp = ((params$n_fft - params$hop_length) / params$n_fft) * 100, wn = window_mapping(params$window), dB = NULL, plot = FALSE)
    # for each column (timestep) of the amplitude spectrogram, extract the centroid
    centroids <- apply(S$amp, 2, function(t) {
       specprop(cbind(S$freq, t), params$sr)$cent
    })
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
   stringsAsFactors = FALSE
)

# compute using soundecology
ais_params <- list(
  acoustic_evenness = list(max_freq = 10000, db_threshold = -47, freq_step = 500),
  bioacoustic_index = list(min_freq = 2000, max_freq = 15000, fft_w = params$n_fft)
)

for (acoustic_index in names(ais_params)) {
    result_path <- file.path(results_path, paste0("tmp_", acoustic_index, ".csv"))
    invisible(capture.output({
      do.call(
        multiple_sounds,
        c(
          list(
            directory = audio_path,
            resultfile = result_path,
            soundindex = acoustic_index,
            no_cores = 1
          ),
          ais_params[[acoustic_index]]
        )
      )
    }))
    result <- read.csv(result_path, stringsAsFactors = FALSE)
    if (!grepl("_index$", acoustic_index)) {
      acoustic_index <- paste0(acoustic_index, "_index")
    }
    df[[acoustic_index]] <- result$LEFT_CHANNEL
}

# cleanup
unlink(results_path, recursive=TRUE)

# persist dataframe
write_parquet(df, output_path)
