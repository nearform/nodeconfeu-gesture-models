rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(pracma)
library(signal)
library(ggplot2)
library(plotly)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(purrr)

bf <- butter(3, 0.1)
dat = read_csv(
    '../data/gestures-v1.csv',
    col_names=c('name', paste0(c('x.', 'y.', 'z.'), rep(1:28, each = 3))),
    col_types=cols(
     .default = col_double(),
     name = col_character()
    )
  ) %>%
  mutate(
    name =unlist(strsplit(name, "(", fixed=T))[seq(1, 2*n(), 2)],
    id = 1:n()
  ) %>%
  gather("time.axis", "acc", -name, -id) %>%
  mutate(
    axis = substring(time.axis, 1, 1),
    sec = (as.integer(substring(time.axis, 3)) - 1) * 0.1
  ) %>%
  select(-time.axis) %>%
  spread(axis, acc) %>%
  filter(!is.na(x) & !is.na(y) & !is.na(z)) %>%
  arrange(id, sec) %>%
  rename(
    x.acc = x,
    y.acc = y,
    z.acc = z
  ) %>%
  mutate(
    x.acc = filtfilt(bf, x.acc),
    y.acc = filtfilt(bf, y.acc),
    z.acc = filtfilt(bf, z.acc),

    x.vel = c(cumtrapz(sec, x.acc)),
    y.vel = c(cumtrapz(sec, y.acc)),
    z.vel = c(cumtrapz(sec, z.acc)),

    x.pos = c(cumtrapz(sec, x.vel)),
    y.pos = c(cumtrapz(sec, y.vel)),
    z.pos = c(cumtrapz(sec, z.vel))
  )


dat.gather = cbind(
  dat %>% select(name, id, sec),
  dat %>% rename(x=x.acc, y=y.acc, z=z.acc) %>% select(x, y, z) %>% gather("axis", "acc", x, y, z),
  dat %>% rename(x=x.vel, y=y.vel, z=z.vel) %>% select(x, y, z) %>% gather("axis", "vel", x, y, z) %>% select(-axis),
  dat %>% rename(x=x.pos, y=y.pos, z=z.pos) %>% select(x, y, z) %>% gather("axis", "pos", x, y, z) %>% select(-axis)
) %>% gather("metric", "value", acc, vel, pos)

p = ggplot(dat.gather, aes(y = value, x = sec, group = as.factor(id), colour=axis)) +
  geom_line(alpha=0.5) +
  facet_grid(metric + axis ~ name)
print(p)

dat %>%
  group_by(id, name) %>%
  summarise(
    x.acc = mean(head(x.acc, 3)),
    y.acc = mean(head(y.acc, 3)),
    z.acc = mean(head(z.acc, 3))
  ) %>%
  group_by(name) %>%
  summarise(
    x.acc.avg = mean(x.acc),
    x.acc.sd = sd(x.acc),
    y.acc.avg = mean(y.acc),
    y.acc.sd = sd(y.acc),
    z.acc.avg = mean(z.acc),
    z.acc.sd = sd(z.acc)
  ) %>%
  print()

dat %>%
  group_by(name, id) %>%
  summarize() %>%
  group_by(name) %>%
  summarize(count = n()) %>%
  print()
