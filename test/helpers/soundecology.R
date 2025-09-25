options(repos = c(CRAN = "https://cloud.r-project.org/"))
if (!requireNamespace("soundecology", quietly = TRUE)) {
    install.packages("soundecology", dependencies = TRUE)
}
if (!requireNamespace("arrow", quietly = TRUE)) {
  install.packages("arrow")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

library(dplyr)
library(arrow)
library(soundecology)

fixtures_path <- "test/fixtures"
audio_path <- file.path(fixtures_path, "audio")
results_path <- file.path(fixtures_path, "results")

dir.create(results_path, recursive = TRUE, showWarnings = FALSE)

# J is the number of seconds
acoustic_indices_params <- list(
  acoustic_complexity = list(max_freq = 24000, fft_w = 1024, j = 5),
  acoustic_evenness = list(max_freq = 24000, db_threshold = -50, freq_step = 1000),
  bioacoustic_index = list(min_freq = 300, max_freq = 24000, fft_w = 1024)
)

# NB: overlap / hop defaults to = 128

df <- NULL

for (acoustic_index in names(acoustic_indices_params)) {
    result_path <- file.path(results_path, paste0("tmp_", acoustic_index, ".csv"))
    params = acoustic_indices_params[[acoustic_index]]

    do.call(
      multiple_sounds,
      c(
        list(
          directory = audio_path,
          resultfile = result_path,
          soundindex = acoustic_index,
          no_cores = 1
        ),
        params
      )
    )

    result <- read.csv(result_path, stringsAsFactors = FALSE)

    if (is.null(df)) {
      df <- result
      colnames(df)[colnames(df) == "LEFT_CHANNEL"] <- acoustic_index
    } else {
      df[[acoustic_index]] <- result$LEFT_CHANNEL
    }
}

df <- df %>% select(-c(SAMPLINGRATE, DURATION, CHANNELS, FFT_W, MIN_FREQ, MAX_FREQ, J, RIGHT_CHANNEL, INDEX, BIT))

colnames(df) <- c(
  "file_name",
  "acoustic_complexity_index",
  "acoustic_evenness_index",
  "bioacoustic_index"
)

parquet_file <- file.path(fixtures_path, "soundecology_indices.parquet")
write_parquet(df, parquet_file)
unlink(results_path, recursive = TRUE)
print(df)
