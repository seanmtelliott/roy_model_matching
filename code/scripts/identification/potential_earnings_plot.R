# Testing some stuff with the PE distribution

pe <- read_csv(file.path(input_dir,"potential_earnings_df.csv"))

firm <- pe %>% 
  select(k_k,s_k,wage_k_k,wage_s_k,k_s,s_s,wage_k_s,wage_s_s) %>%
  mutate(output = wage_k_k + wage_s_s,
         inequality = wage_k_k-wage_s_s)

firm_types <- 