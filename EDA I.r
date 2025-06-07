
library(readxl)
library(dplyr)
library(tidyr)
library(janitor)
library(ggplot2)
library(scales) 
library(lubridate)


nodes  <- read_excel("C:/Users/Zenbook/Downloads/data_Maastricht_2024 (1).xlsx", sheet = "Nodes")
edges  <- read_excel("C:/Users/Zenbook/Downloads/data_Maastricht_2024 (1).xlsx", sheet = "Edges")
cbs_squares <- read_excel(
  "C:/Users/Zenbook/Downloads/Copy of CBS_Squares_cleaned(1).xlsx",
  skip  = 1
) %>% clean_names()
service_points <- read_excel("C:/Users/Zenbook/Downloads/data_Maastricht_2024 (1).xlsx", sheet = "Service Point Locations")
at_home <- read_excel("C:/Users/Zenbook/Downloads/data_Maastricht_2024 (1).xlsx", sheet = "At-home Deliveries")
picked_up <- read_excel("C:/Users/Zenbook/Downloads/data_Maastricht_2024 (1).xlsx", sheet = "Service Point Parcels Picked Up")
cbs_squares_clean <- read_excel("C:/Users/Zenbook/Downloads/data_Maastricht_2024 (1).xlsx", sheet = "CBS Squares Clean")


##_________________________________
## Total Pickups per Service Point
##_________________________________

service_points %>%
  ggplot(aes(x = reorder(`Location ID`, -`Total Pickups`), y = `Total Pickups`)) +
  geom_col(fill = "dodgerblue", color = "black", linewidth = 0.2) +
  labs(
    title = "Total Pickups per Service Point",
    x = "Service Point",
    y = "Total Pickups"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.title = element_text(face = "bold"))


##________________________________
## Pickup Rates per Service Point
##________________________________

service_points %>%
  mutate(PickupRate = `Total Pickups` / (`Total Pickups` + `Total Deliveries`)) %>%
  ggplot(aes(x = reorder(`Location ID`, -PickupRate), y = PickupRate)) +
  geom_col(fill = "deepskyblue3", color = "black", linewidth = 0.2) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +  
  labs(
    title = "Pickup Rate per Service Point",
    x = "Service Point",
    y = "Pickup Rate"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme(plot.title = element_text(face = "bold"))


##____________________
## Square Populations
##____________________

cbs_squares %>%
  ggplot(aes(x = population, fill = ..count..)) +
  geom_histogram(bins = 20, color = "black", linewidth = 0.2) +
  scale_fill_viridis_c(name = "Count") +
  labs(
    title = "Square Populations",
    x = "Population per Square", y = "Number of Squares"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
  ) 


##____________________
## Working Population
##____________________

cbs_squares <- cbs_squares %>%
  mutate(
    working_population = age15_24 + age25_44 + age45_64
  )

summary(cbs_squares$working_population)

top10_squares <- cbs_squares %>%
  arrange(desc(working_population)) %>%
  slice_head(n = 10)

ggplot(top10_squares, aes(
  x = reorder(square, working_population), 
  y = working_population
)) +
  geom_col(fill = "cadetblue3", color = "black", linewidth = 0.2) +
  coord_flip() +
  labs(
    title = "Top 10 CBS Squares by Working Population (Age 15–64)",
    x = "CBS Square",
    y = "Working Population"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))


##______________________________
## Picked Up vs At Home
##______________________________

picked_up2 <- picked_up %>%
  rename(day = `day\\location`) %>%
  pivot_longer(cols = -day, names_to = "location", values_to = "parcels") %>%
  mutate(
    type = "Picked Up",
    day = as.integer(day),
    date = as.Date("2024-01-01") + days(day - 1),
    month = month(date, label = TRUE)
  )

at_home2 <- at_home %>%
  rename(day = `day\\location`) %>%
  pivot_longer(cols = -day, names_to = "location", values_to = "parcels") %>%
  mutate(
    type = "At Home",
    day = as.integer(day),
    date = as.Date("2024-01-01") + days(day - 1),
    month = month(date, label = TRUE)
  )

combined <- bind_rows(picked_up2, at_home2)
combined %>%
  group_by(month, type) %>%
  summarise(total_parcels = sum(parcels), .groups = "drop") %>%
  ggplot(aes(x = month, y = total_parcels, fill = type)) +
  geom_col(position = "dodge") +
  labs(
    title = "Picked Up vs At-Home Deliveries by Month",
    x = "Month",
    y = "Total Parcels",
    fill = "Delivery Type"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))


