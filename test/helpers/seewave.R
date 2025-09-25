if (!requireNamespace("seewave", quietly = TRUE)) {
  install.packages("seewave")
}
if (!requireNamespace("tuneR", quietly = TRUE)) {
  install.packages("tuneR")
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
})

fixtures_path <- "test/fixtures"
audio_path <- file.path(fixtures_path, "audio")
results_path <- file.path(fixtures_path, "results")

params <- yaml::read_yaml(file.path(fixtures_path, "fft_params.yml"))

window_length <- params$window_length
window <- params$window
sample_rate <- params$sample_rate

dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

files <- list.files(audio_path, pattern = "\\.wav$", full.names = TRUE)

aci = sapply(lapply(files, readWave), function(wav) {
    ACI(wav, sample_rate, channel = 1, wl = window_length, ovlp = 0, wn = window, nbwindows = 1)
})
df <- data.frame(
   file_name = basename(files),
   acoustic_complexity_index = aci,
   stringsAsFactors = FALSE
)
parquet_file <- file.path(fixtures_path, "indices.parquet")
write_parquet(df, parquet_file)
