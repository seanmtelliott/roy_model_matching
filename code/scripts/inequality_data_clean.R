# Cleaning up the Bloom data

ineq_data <- read_csv(file.path(input_dir,"FUI_data.csv"),show_col_types = F)

# Convert it to long format

ineq_long <- ineq_data %>% pivot_longer(cols = p1:p100,names_to = "percentile",values_to = "income")

# Set up firm/ind as variables

ineq <- ineq_long %>% pivot_wider(names_from = Level,values_from = income) %>%
  mutate(ln_firm = log(firm),
         ln_ind = log(individual),
         ln_within = ln_ind - ln_firm) %>%
  select(-c(Description,`Pct of distribution`,`Long description`))

# Test to see what the inequality plots look like

year_sub1 <- ineq %>% filter(Year == 1983)
year_sub2 <- ineq %>% filter(Year == 2013)

year_compare <- year_sub2 %>%
  left_join(year_sub1,by="percentile",suffix = c("_2","_1")) %>%
  mutate(ind_ineq = ln_ind_2 - ln_ind_1,
         firm_ineq = ln_firm_2 - ln_firm_1,
         within_ineq = ln_within_2 - ln_within_1) %>%
  select(c(percentile,ind_ineq,firm_ineq,within_ineq)) %>%
  pivot_longer(cols = ind_ineq:within_ineq,names_to = "type",values_to = "diff") %>%
  rowwise() %>%
  mutate(percentile = as.numeric(strsplit(percentile,"p")[[1]][2])) %>%
  filter(percentile != 100)

ggplot(data=year_compare) + aes(x=percentile,y=diff,color=type) + geom_line()

year_compare_wide <- year_sub2 %>%
  left_join(year_sub1,by="percentile",suffix = c("_2","_1")) %>%
  mutate(ind_ineq = ln_ind_2 - ln_ind_1,
         firm_ineq = ln_firm_2 - ln_firm_1,
         within_ineq = ln_within_2 - ln_within_1) %>%
  select(c(percentile,ind_ineq,firm_ineq,within_ineq))  %>%
  rowwise() %>%
  mutate(percentile = as.numeric(strsplit(percentile,"p")[[1]][2]))
