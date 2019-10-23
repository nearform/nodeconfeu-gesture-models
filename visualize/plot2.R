rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(pracma)
library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)

dat = bind_rows(
    conor = read_csv('./data/conor.csv') %>% mutate(id = paste0(id, '.conor')),
    james = read_csv('./data/james.csv') %>% mutate(id = paste0(id, 'james')),
    .id = "person"
  ) %>%
  group_by(subset, label, id, person, dimension) %>%
  mutate(
    velocity = cumtrapz(time, acceleration),
    position = cumtrapz(time, velocity)
  )

dat.plot = dat %>%
  gather("metric", "value", acceleration, velocity, position)

p = ggplot(dat.plot, aes(y = value, x = time, group = as.factor(id), colour=person)) +
  geom_hline(yintercept=0) +
  geom_line(alpha=0.1) +
  facet_grid(metric + dimension ~ label) +
  guides(colour = guide_legend(override.aes = list(alpha = 1)))
print(p)
