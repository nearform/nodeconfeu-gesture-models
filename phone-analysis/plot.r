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
dat = read_csv('data/square-floor-100hz.csv') %>%
  mutate(
    sec = (ts - first(ts)) / 1000,

    # x.acc = x - mean(x[sec < 10]),
    # y.acc = y - mean(y[sec < 10]),
    # z.acc = z - mean(z[sec < 10])

    x.acc = filtfilt(bf, x - mean(x[sec < 10])),
    y.acc = filtfilt(bf, y - mean(y[sec < 10])),
    z.acc = filtfilt(bf, z - mean(z[sec < 10]))

    #x.acc = ifelse(abs(x - mean(x[sec < 10])) < 1.96 * sd(x[sec < 10]), 0, x - mean(x[sec < 10])),
    #y.acc = ifelse(abs(x - mean(y[sec < 10])) < 1.96 * sd(y[sec < 10]), 0, y - mean(y[sec < 10])),
    #z.acc = ifelse(abs(x - mean(z[sec < 10])) < 1.96 * sd(z[sec < 10]), 0, z - mean(z[sec < 10]))
  ) %>%
  filter(sec > 10 & sec < 10 + 20) %>%
  select(-x, -y, -z, -gx, -gy, -gz, -ts) %>%
  mutate(
    sec = sec - 10,

    x.vel = c(cumtrapz(sec, x.acc)),
    y.vel = c(cumtrapz(sec, y.acc)),
    z.vel = c(cumtrapz(sec, z.acc)),

    x.pos = c(cumtrapz(sec, x.vel)),
    y.pos = c(cumtrapz(sec, y.vel)),
    z.pos = c(cumtrapz(sec, z.vel))
  )
plot.range = with(dat, range(c(x.pos, y.pos, z.pos)))

dat.gather = cbind(
  dat %>% select(sec),
  dat %>% rename(x=x.acc, y=y.acc, z=z.acc) %>% select(x, y, z) %>% gather("axis", "acc", x, y, z),
  dat %>% rename(x=x.vel, y=y.vel, z=z.vel) %>% select(x, y, z) %>% gather("axis", "vel", x, y, z) %>% select(-axis),
  dat %>% rename(x=x.pos, y=y.pos, z=z.pos) %>% select(x, y, z) %>% gather("axis", "pos", x, y, z) %>% select(-axis)
) %>% gather("metric", "value", acc, vel, pos)

p = ggplot(dat.gather, aes(y = value, x = sec, colour = axis)) +
  geom_line() +
  facet_grid(axis ~ metric)
print(p)

# p = ggplot(dat, aes(y = y.pos, x = x.pos, colour=sec)) +
#   geom_point()
# print(p)

p = plot_ly(dat, x=~x.pos, y=~y.pos, z=~z.pos, color=~sec) %>%
  add_markers() %>%
  layout(title = "Position estimate",
         scene = list(
           xaxis=list(range=plot.range),
           yaxis=list(range=plot.range),
           zaxis=list(range=plot.range)
         )
  )
print(p)

