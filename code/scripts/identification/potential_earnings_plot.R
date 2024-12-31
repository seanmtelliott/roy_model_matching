# Testing some stuff with the PE distribution

pe <- read_csv(file.path(input_dir,"potential_earnings_df.csv"))

firm <- pe %>% 
  select(k_k,s_k,wage_k_k,wage_s_k,k_s,s_s,wage_k_s,wage_s_s) %>%
  mutate(output = wage_k_k + wage_s_s,
         inequality = wage_k_k-wage_s_s)

firm_types <- firm %>%
  mutate(firmbin = round(output,digits=1),
         ineqbin = round(inequality,digits=1),
         k_k_bin = round(wage_k_k,digits=2),
         s_s_bin = round(wage_s_s,digits=2))

bins <- sort(unique(firm_types$firmbin))

for(b in bins){
  
  bin_sub <- firm_types %>% filter(firmbin==b)
  
  skills_df <- bin_sub %>% 
    select(k_k,s_k,k_s,s_s) %>%
    pivot_longer(cols=k_k:s_s,names_to = "type",values_to = "skill")
  
  skill_density <- ggplot(skills_df, aes(x=skill, color=type)) +
    geom_density()
  
  
  ggsave(file.path(output_dir,"potential_earnings",paste("output_plot_",b,".png",sep="")),skill_density)
  
}

skills_df_all <- firm_types %>% 
  select(k_k,s_k,k_s,s_s) %>%
  pivot_longer(cols=k_k:s_s,names_to = "type",values_to = "skill")

skill_density_all <- ggplot(skills_df_all, aes(x=skill, color=type)) +
  geom_density()

ggsave(file.path(output_dir,"potential_earnings",paste("output_plot_","comb",".png",sep="")),skill_density)