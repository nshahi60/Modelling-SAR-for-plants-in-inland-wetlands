# Code to build a global species-area relationship for plants based on multiple input data sources
# September 2024

# Set working directory
setwd("C://Users/aafkes/Documents/Projects/Nicole/SARs/R")

# Enable packages
library(lme4) # for linear mixed effects modelling
library(MuMIn) # for marginal and conditional R^2
library(ggplot2) # for plotting

# Read data (I created a combined file in which the original datasets are coded by country using ISO3 codes)
df <- read.csv("data_combined.csv", head = T, stringsAsFactors = T)

# Log-transform area and species richness
df[, c("area_km2", "species_richness")] <- log(df[, c("area_km2", "species_richness")], 10)

# Fit species-area relationship with random slope and intercept according to dataset ID
SAR_model <- lmer(species_richness ~ area_km2 + (area_km2|dataset), data = df)
summary(SAR_model)

# Retrieve R-squared values (marginal and conditional)
r.squaredGLMM(SAR_model) 

# Inspect residuals
hist(resid(SAR_model)) 
qqnorm(resid(SAR_model))
qqline(resid(SAR_model))
plot(predict(SAR_model), resid(SAR_model))
abline(h=0)

# Create predictions based on the fixed effect of area (ignoring random slopes)
fitted_SAR <- predict(SAR_model, newdata = df, re.form = ~0)

# Visualize the results
p <- ggplot(df, aes(x = area_km2, y = species_richness, colour = dataset)) +
  geom_point(size = 1) +
  geom_line(aes(y = predict(SAR_model)), linewidth = 1) +
  geom_line(aes(y = fitted_SAR), linewidth = 1, colour = "black") +
  theme_bw()
p + xlab("log(area)") + ylab("log(species richness)")