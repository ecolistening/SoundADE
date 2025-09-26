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

fixtures_path <- args[1]
output_file <- args[2]
audio_path <- file.path(fixtures_path, "audio")
results_path <- file.path(fixtures_path, "results")

params <- yaml::read_yaml(file.path(fixtures_path, "audio_params.yml"))

n_fft <- params$n_fft
hop <- params$hop_length
window <- params$window
sr <- params$sr
flim <- params$flim
# calculate the overlap in %
ovlp = ((n_fft - hop) / n_fft) * 100

# for testing, we re-use the params file
# but seewave uses "hanning" instead of "hann" as the window argument
window_mapping <- function(x) {
  mapping <- c(
     "hann" = "hanning"
  )
  ifelse(x %in% names(mapping), mapping[x], x)
}

dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

files <- list.files(audio_path, pattern = "\\.wav$", full.names = TRUE)

# compute using seewave
wavs = lapply(files, readWave)

rmss = sapply(wavs, function(wav) {
    samples <- wav@left
    mean(rms(samples, wl = n_fft))
})

scs = sapply(wavs, function(wav) {
    S <- spectro(wav, sr, channel = 1, wl = n_fft, ovlp = ovlp, wn = window_mapping(window), dB = NULL)

    centroids <- apply(S$amp, 2, function(t) {
       specprop(cbind(S$freq, t), sr)$cent
    })

    mean(centroids, na.rm = TRUE)
})

acis = sapply(wavs, function(wav) {
    ACI(wav, sr, channel = 1, wl = n_fft, ovlp = ovlp, wn = window_mapping(window), nbwindows = 1)
})

sfs = sapply(wavs, function(wav) {
    mean(specflux(wav, sr, channel = 1, wl = n_fft, ovlp = ovlp, wn = window_mapping(window), norm = TRUE, p = 2)[,2])
})

zcrs = sapply(wavs, function(wav) {
    mean(zcr(wav, channel = 1, sr, wl = n_fft, ovlp = ovlp)[,2])
})

df <- data.frame(
   file_name = basename(files),
   spectral_centroid = scs,
   acoustic_complexity_index = acis,
   zero_crossing_rate = zcrs,
   spectral_flux = sfs,
   root_mean_square = rmss,
   stringsAsFactors = FALSE
)

# compute using soundecology
ais_params <- list(
  acoustic_evenness = list(max_freq = sr / 2, db_threshold = -50, freq_step = 500),
  bioacoustic_index = list(min_freq = flim[[1]], max_freq = sr / 2, fft_w = n_fft)
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
    unlink(result_path)
}

# persist dataframe
parquet_file <- file.path(fixtures_path, output_file)
write_parquet(df, parquet_file)
