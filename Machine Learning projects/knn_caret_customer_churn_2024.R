###############################################################################
# Code written by Jessica Hooker (January 2024)

# Brief description of the data:
## The data set is practice data set comprising 10,000 rows with 50 variables. 
## The data is from a 'telecommunication company' and includes variables   
## related to customer demographics, services utilized, service usage, and
## responses to a customer satisfaction survey.

# The primary goal of this analysis was to building a k-nearest neighbors model
## that accurately predicted whether a customer had churned from key 
## customer and services characteristics. 

## This file comprises the syntax for the data preprocessing and model building. 



###############################################################################


# load in libraries
library(readr)
library(tidyverse)
library(naniar)
library(knitr)
library(ggpubr)
library(psych)
library(caret)
library(polycor)
library(pROC)

###### Data Preprocessing 

# load in the medical readmission data set 
churn_df <- read_csv('churn.csv')

# get information about df
glimpse(churn_df)

# Data Cleaning  -----------------------------------------

# Check for Duplicates

## Full Duplicates
sum(duplicated(churn_df))

## Duplicate Patients (Wickham et al., n.d.) 
churn_df %>% count(Customer_id) %>% 
  filter(n>1)


# Check Data Formatting
glimpse(churn_df)

## round quantitative variables (Zach, 2023)
churn_df <- churn_df %>% mutate(
  across(c(Income, Outage_sec_perweek, Tenure, MonthlyCharge, Bandwidth_GB_Year), 
         \(x) round(x, 2))
)

## Verify Data formatting
glimpse(churn_df)


# Check for Missing Data (Batra et al., 2021; Tierney, 2023)

## search hidden missing data
print(churn_df %>% miss_scan_count(search = list("Na", "N/A","N/a", 
                                               "NULL", "null")
                                 ), n = 50
      )

## Overall Missing Data
n_miss(churn_df)


# Exploratory Data Analysis -----------------------------

# Summary Statistics

## quantitative variables

churn_df %>% 
  select(Age, Income, Outage_sec_perweek, Email, Contacts, Yearly_equip_failure,
         Tenure, MonthlyCharge, 
         Bandwidth_GB_Year) %>% 
  describe()

# qualitative variables

## yes/no variables 
churn_df %>% 
  select(Churn, Techie, Port_modem, Tablet, Phone, Multiple, OnlineSecurity,
         OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
         StreamingMovies, PaperlessBilling) %>% 
  map(table, useNA = 'ifany') %>% 
  imap(~as.list(.x) %>% as.data.frame %>% 
         mutate(Variable = .y) %>% relocate(Variable)) %>% 
  bind_rows %>% 
  as_tibble() %>% 
  knitr::kable(format = "simple",caption = "Sample Counts for Yes/No Variables")

churn_df %>% 
  select(Churn, Techie, Port_modem, Tablet, Phone, Multiple, OnlineSecurity,
         OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
         StreamingMovies, PaperlessBilling) %>% 
  map(table, useNA = 'ifany') %>% 
  map(prop.table) %>% map(round, digits = 2) %>% 
  imap(~as.list(.x) %>% as.data.frame %>% 
         mutate(Variable = .y) %>% relocate(Variable)) %>% 
  bind_rows %>% 
  as_tibble() %>% 
  knitr::kable(format = "simple",caption = "Sample Proportions for Yes/No Variables")


## other qualitative vars (Devlin, 2015)

churn_df %>% 
  count(Contract) %>% 
  mutate(proportion = round(prop.table(n), 2)) %>% 
  bind_rows %>% 
  as_tibble() %>% 
  knitr::kable(format = "simple", caption = "Count and Proportions for Contract")


churn_df %>% 
  count(InternetService) %>% 
  mutate(proportion = round(prop.table(n), 2)) %>% 
  bind_rows %>% 
  as_tibble() %>% 
  knitr::kable(format = "simple", 
               caption = "Count and Proportions for InternetService")

churn_df %>% 
  count(PaymentMethod) %>% 
  mutate(proportion = round(prop.table(n), 2)) %>% 
  bind_rows %>% 
  as_tibble() %>% 
  knitr::kable(format = "simple", 
               caption = "Count and Proportions for PaymentMethod")

# Univariate Visualizations & Outliers

#Histograms

age_hist <- ggplot(churn_df, aes(Age))+
  geom_histogram(bins = 40, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Age (years)", 
       y = "Count") +
  theme_classic() + 
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

income_hist <- ggplot(churn_df, aes(Income))+
  geom_histogram(bins = 40, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Income ($)", 
       y = "Count") +
  theme_classic() + 
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

outage_hist <- ggplot(churn_df, aes(Outage_sec_perweek))+
  geom_histogram(bins = 40, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Outage Seconds Per Week", 
       y = "Count") +
  theme_classic() + 
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

email_hist <- ggplot(churn_df, aes(Email))+
  geom_histogram(bins = 30, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Emails in Last Year", y = "Count") +
  theme_classic() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05)))

contacts_hist <- ggplot(churn_df, aes(Contacts))+ 
  geom_histogram(bins = 30, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Technical Support Contacts", y = "Count") +
  theme_classic() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

equip_hist <- ggplot(churn_df, aes(Yearly_equip_failure))+
  geom_histogram(bins = 40, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Equipment Failures in Last Year", 
       y = "Count") +
  theme_classic() + 
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

tenure_hist <- ggplot(churn_df, aes(Tenure))+
  geom_histogram(bins = 40, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Tenure (months)", 
       y = "Count") +
  theme_classic() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

charge_hist <- ggplot(churn_df, aes(MonthlyCharge))+
  geom_histogram(bins = 40, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Average Monthly Charge", 
       y = "Count") +
  theme_classic() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

bandwidth_hist <- ggplot(churn_df, aes(Bandwidth_GB_Year))+
  geom_histogram(bins = 40, fill = "deepskyblue4", color = "gray80")+
  labs(x = "Average Bandwidth Per Year (GB)", 
       y = "Count") +
  theme_classic() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.01)))

# combine plots (kassambara, 2017)
hist_fig <- ggarrange(age_hist, income_hist, outage_hist, email_hist, 
                      contacts_hist, equip_hist, tenure_hist,
                   charge_hist, bandwidth_hist,
                   ncol = 3, 
                   nrow = 3)
hist_fig
annotate_figure(hist_fig,
                top = text_grob("Histograms of Quantitative Variables", 
                                face = "bold", size = 14)
)

# box plots (Wickam et al., n.d.) 

age_box <- ggplot(churn_df, aes(Age))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Age, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

income_box <- ggplot(churn_df, aes(Income))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Income, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

outage_box <- ggplot(churn_df, aes(Outage_sec_perweek))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Outage_sec_perweek, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

email_box <- ggplot(churn_df, aes(Email))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Email, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

contacts_box <- ggplot(churn_df, aes(Contacts))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Contacts, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )
  
equip_box <- ggplot(churn_df, aes(Yearly_equip_failure))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Yearly_equip_failure, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )
  
tenure_box <- ggplot(churn_df, aes(Tenure))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Tenure, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

charge_box <- ggplot(churn_df, aes(MonthlyCharge))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(MonthlyCharge, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )
  
bandwidth_box <- ggplot(churn_df, aes(Bandwidth_GB_Year))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(round(after_stat(xmin), 3), round(after_stat(xmax), 3)),
    x = stage(Bandwidth_GB_Year, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

# combine plots (kassambara, 2017)
box_fig <- ggarrange(age_box, income_box, outage_box, email_box, 
                      contacts_box, equip_box, tenure_box,
                      charge_box, bandwidth_box,
                      ncol = 3, 
                      nrow = 3)
box_fig
annotate_figure(box_fig,
                top = text_grob("Box Plots of Quantitative Variables", 
                                face = "bold", size = 14)
)

# address outliers

## outage_ser_perweek
### segment outliers
outage_outliers <- churn_df[which(churn_df$Outage_sec_perweek < 2.10 |
                                    churn_df$Outage_sec_perweek >17.86),]

### number of outliers
nrow(outage_outliers)

#### view range of values (Venkatachalam, n.d.)
outage_outliers %>% 
  arrange(desc(Outage_sec_perweek)) %>%
  select(Outage_sec_perweek) %>% 
  headTail(top = 5, bottom = 5)

### replace outliers
churn_df[which(churn_df$Outage_sec_perweek < 2.10),]$Outage_sec_perweek <- 2.10
churn_df[which(churn_df$Outage_sec_perweek > 17.86),]$Outage_sec_perweek <- 17.86


### confirm (Wickam et al., n.d.)
ggplot(churn_df, aes(Outage_sec_perweek))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Outage_sec_perweek, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

# emails
### segment outliers
email_outliers <- churn_df[which(churn_df$Email < 4 |
                                   churn_df$Email > 20),]

### number of outliers
nrow(email_outliers)

### view range of values (Venkatachalam, n.d.)
email_outliers %>% 
  arrange(desc(Email)) %>%
  select(Email) %>% 
  headTail(top = 5, bottom = 5)

### replace outliers
churn_df[which(churn_df$Email < 4),]$Email <- 4
churn_df[which(churn_df$Email > 20),]$Email <- 20

#### confirm treatment (Wickam et al., n.d.)
ggplot(churn_df, aes(Email))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Email, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

# Contacts 
### segment outliers
contacts_outliers <- churn_df[which(churn_df$Contacts > 5),]

### number of outliers
nrow(contacts_outliers)

#### view range of values (Venkatachalam, n.d.)
contacts_outliers %>% 
  arrange(desc(Contacts)) %>%
  select(Contacts) %>% 
  headTail(top = 5, bottom = 5)

### replace outliers
churn_df[which(churn_df$Contacts > 5),]$Contacts <- 5

##  confirm values were replaced
sort(unique(churn_df$Contacts))

#### confirm treatment (Wickam et al., n.d.)
ggplot(churn_df, aes(Contacts))+
  geom_boxplot(outlier.colour = "blue") +
  geom_text(aes(
    label = c(after_stat(xmin), after_stat(xmax)),
    x = stage(Contacts, after_stat = c(xmin,xmax))),
    stat = "boxplot",
    vjust = -0.5
  )

# Yearly Equipment Failure
### segment outliers
equip_failure_outliers <- churn_df[which(churn_df$Yearly_equip_failure > 2),]

#### number of outliers
nrow(equip_failure_outliers)

#### view range of values (Venkatachalam, n.d.)
equip_failure_outliers %>% 
  arrange(desc(Yearly_equip_failure)) %>%
  select(Yearly_equip_failure) %>% 
  headTail(top = 5, bottom = 5)

### replace outliers
churn_df[which(churn_df$Yearly_equip_failure > 2),]$Yearly_equip_failure <- 2

### factor variable
churn_df<- churn_df %>% 
  mutate(
    Yearly_equip_failure = factor(Yearly_equip_failure, levels = c(0:2), 
                                  labels = c("0", "1", "2 or more")
    ),
    
  )

### confirm 
is.factor(churn_df$Yearly_equip_failure)
levels(churn_df$Yearly_equip_failure)

churn_df %>% 
  count(Yearly_equip_failure) %>% 
  mutate(proportion = round(prop.table(n), 2)) %>% 
  bind_rows %>% 
  as_tibble() %>% 
  knitr::kable(format = "simple", 
               caption = "Count and Proportions for Yearly_equip_failure")

## yes/no variables 
## qualitative variables (sauer, 2016)

### yes/no variables  
# dependent variable
ggplot(churn_df, aes(Churn))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Churn", 
       y = "Count",
       title = "Bar Plot of Churn Status") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))


#independent variables
techie_bar <- ggplot(churn_df, aes(Techie))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Self-reported Techie", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

modem_bar <- ggplot(churn_df, aes(Port_modem))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Portable Modem", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

tablet_bar <- ggplot(churn_df, aes(Tablet))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has a Tablet", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic()  +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

phone_bar <- ggplot(churn_df, aes(Phone))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Phone Service", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0))) 

multiple_bar <- ggplot(churn_df, aes(Multiple))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Multiple Lines", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

security_bar <- ggplot(churn_df, aes(OnlineSecurity))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Online Security Add-on", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0))) 

backup_bar <- ggplot(churn_df, aes(OnlineBackup))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Online Backup Add-on", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))  

device_bar <- ggplot(churn_df, aes(DeviceProtection))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Device Protection Add-on", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))  

support_bar <- ggplot(churn_df, aes(TechSupport))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Technical Support Add-on", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))  

tv_bar <- ggplot(churn_df, aes(StreamingTV))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has TV Streaming", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))  

movies_bar <- ggplot(churn_df, aes(StreamingMovies))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Movie Streaming", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

billing_bar <- ggplot(churn_df, aes(PaperlessBilling))+
  geom_bar(fill = "midnightblue" )+
  labs(x = "Has Paperless Billing", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = 1, 
            fontface = "bold", color = "gray95") +
  theme_classic() +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

##### combine plots (kassambara, 2017)
bars_y_n<- ggarrange(techie_bar , modem_bar, 
                   tablet_bar, phone_bar,
                   multiple_bar, security_bar, 
                   backup_bar, device_bar, 
                   support_bar, tv_bar,
                   movies_bar, billing_bar,
                   ncol = 3, 
                   nrow = 4)
bars_y_n
annotate_figure(bars_y_n,
                top = text_grob("Bar Plots of Yes/No Variables", 
                                face = "bold", size = 14)
)

### other categorical variables (sauer, 2016)
equip_bar <- ggplot(churn_df, aes(Yearly_equip_failure))+
  geom_bar(fill = "darkred" )+
  labs(x = "Yearly Amount of Equipment Failure", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.5, 
            fontface = "bold") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

contract_bar <- ggplot(churn_df, aes(Contract))+
  geom_bar(fill = "darkred" )+
  labs(x = "Contract Type", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.5, 
            fontface = "bold") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))+
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

internet_bar <- ggplot(churn_df, aes(InternetService))+
  geom_bar(fill = "darkred" )+
  labs(x = "Type of Internet Service", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.5, 
            fontface = "bold") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

payment_bar <- ggplot(churn_df, aes(PaymentMethod))+
  geom_bar(fill = "darkred" )+
  labs(x = "Payment Method", 
       y = "Count") +
  geom_text(aes(label=scales::percent(after_stat(count)/sum(after_stat(count)), 0.1)), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.5, 
            fontface = "bold") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

##### combine plots (kassambara, 2017)
bars_other <- ggarrange(equip_bar, contract_bar, internet_bar, payment_bar,
                     ncol = 2, 
                     nrow = 2)
bars_other
annotate_figure(bars_other,
                top = text_grob("Bar Plots of Other Qualitative Variables", 
                                face = "bold", size = 14)
)
## bivariate visualizations

## quantitative
age_density <- ggplot(churn_df, aes(x = Age, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Customer Age (Years",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))

income_density <- ggplot(churn_df, aes(x = Income, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Income ($)",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))

outage_density <- ggplot(churn_df, aes(x = Outage_sec_perweek, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Outage Seconds Per Week",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))

email_density <- ggplot(churn_df, aes(x = Email, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Emails in Last Year",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))

support_density <- ggplot(churn_df, aes(x = Contacts, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Technical Support Contacts",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))

tenure_density <- ggplot(churn_df, aes(x = Tenure, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Tenure (months)",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))

charge_density <- ggplot(churn_df, aes(x = MonthlyCharge, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Average Monthly Charge",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))

bandwidth_density <- ggplot(churn_df, aes(x = Bandwidth_GB_Year, color = Churn, fill= Churn))+
  geom_density(alpha = 0.4, lwd = 1.2) +
  scale_color_manual(values = c("purple4", "lightgoldenrod"))+
  scale_fill_manual(values = c("purple4", "lightgoldenrod"))+
  labs(x = "Average Bandwidth Per Year (GB)",
       y = "Density",
       color = "Churn Status",
       fill = "Churn Status") +
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  scale_y_continuous(expand = expansion(mult = c(0, .05))) +
  scale_x_continuous(expand = expansion(mult = c(0.005, 0.005)))


##### combine plots (kassambara, 2017)
biv_density <- ggarrange(age_density, income_density, outage_density , email_density, 
                    support_density, tenure_density,
                    charge_density,bandwidth_density, 
                    ncol = 3, 
                    nrow = 3)
biv_density
annotate_figure(biv_density,
                top = text_grob("Density Plots of Quantitative Variables by Churn Status", 
                                face = "bold", size = 14)
)


# yes/no variables (rishabhchakrabortygfg, 2021; sauer, 2016)
techie_biv_bar <- ggplot(churn_df, aes(Techie, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Self-reported Techie", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

modem_biv_bar <- ggplot(churn_df, aes(Port_modem, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Portable Modem", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

tablet_biv_bar <- ggplot(churn_df, aes(Tablet, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has a Tablet", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

phone_biv_bar <- ggplot(churn_df, aes(Phone, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Phone Service", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

multiple_biv_bar <- ggplot(churn_df, aes(Multiple, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Multiple Lines", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

security_biv_bar <- ggplot(churn_df, aes(OnlineSecurity, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Online Security Add-on", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

backup_biv_bar <- ggplot(churn_df, aes(OnlineBackup, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Online Backup Add-on", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

device_biv_bar <- ggplot(churn_df, aes(DeviceProtection, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Device Protection Add-on", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

support_biv_bar <- ggplot(churn_df, aes(TechSupport, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Technical Support Add-on", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

tv_biv_bar <- ggplot(churn_df, aes(StreamingTV, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has TV Streaming", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

movie_biv_bar <- ggplot(churn_df, aes(StreamingMovies, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Movie Streaming", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

billing_biv_bar <- ggplot(churn_df, aes(PaperlessBilling, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Has Paperless Billing", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

##### combine plots (kassambara, 2017)
biv_y_n<- ggarrange(techie_biv_bar , modem_biv_bar, 
                   tablet_biv_bar, phone_biv_bar,
                   multiple_biv_bar, security_biv_bar, 
                   backup_biv_bar, device_biv_bar, 
                   support_biv_bar, tv_biv_bar,
                   movie_biv_bar, billing_biv_bar,
                   ncol = 3, 
                   nrow = 4)
biv_y_n
annotate_figure(biv_y_n,
                top = text_grob("Bar Plots of Yes/No Variables by Churn Status", 
                                face = "bold", size = 14),
                bottom = text_grob("Note: Percent values represent conditional proportions of Churn across each level within variables.",
                                   size = 10)
)



# other qualitative variables

equip_biv_bar <- ggplot(churn_df, aes(Yearly_equip_failure, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Yearly Amount of Equipment Failure", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

contract_biv_bar <- ggplot(churn_df, aes(Contract, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Contract Type", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

service_biv_bar <- ggplot(churn_df, aes(InternetService, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "Internet Service Type", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  theme(legend.position = "bottom"  ) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

payment_biv_bar <- ggplot(churn_df, aes(PaymentMethod, fill = Churn))+
  geom_bar(position = position_dodge()) +
  geom_text(aes(label=scales::percent(after_stat(count)/tapply(after_stat(count), x ,sum)[x]) ), 
            stat = "count", 
            position = position_dodge(width = 1), vjust = -.6,
            fontface = "bold") +
  scale_fill_manual(values = c("cadetblue", "violetred4")) +
  labs(x = "PaymentMethod", y = "Count", color = "Churn Status",
       fill = "Churn Status") + 
  theme_classic() +
  # (Alboukadel, 2018)
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, vjust = .60)) +
  coord_cartesian(ylim = c(0, 10000))  +
  scale_y_continuous(expand = expansion(mult = c(0, 0)))

##### combine plots (kassambara, 2017)
biv_bars<- ggarrange(equip_biv_bar , contract_biv_bar, 
                   service_biv_bar, payment_biv_bar,
                   ncol = 2, 
                   nrow = 2)
biv_bars
annotate_figure(biv_bars,
                top = text_grob("Bar Plots of Categocial Variables by Churn Status", 
                                face = "bold", size = 14),
                bottom = text_grob("Note: Percent values represent conditional proportions of Churn across each level within variables.",
                                   size = 10)
)


# Data Transformation -----------------------------------

# Reduce Data Set to model variables
churn_df_init <- churn_df %>% 
  select(Customer_id, Churn, Age, Income, Outage_sec_perweek, Email,
         Contacts, Tenure, MonthlyCharge, Bandwidth_GB_Year, Techie, Port_modem, 
         Tablet, Phone, Multiple,OnlineSecurity, OnlineBackup, DeviceProtection, 
         TechSupport,StreamingTV, StreamingMovies, PaperlessBilling, 
         Yearly_equip_failure, Contract, InternetService, PaymentMethod)

glimpse(churn_df_init)

# format nominal variables into factors
churn_df_init <- churn_df_init %>% mutate(
  Churn = factor(Churn),
  Techie = factor(Techie), 
  Port_modem = factor(Port_modem),
  Tablet = factor(Tablet),
  Phone = factor(Phone),
  Multiple = factor(Multiple),
  OnlineSecurity = factor(OnlineSecurity),
  OnlineBackup = factor(OnlineBackup),
  DeviceProtection = factor(DeviceProtection),
  TechSupport = factor(TechSupport),
  StreamingTV = factor(StreamingTV),
  StreamingMovies = factor(StreamingMovies),
  PaperlessBilling = factor(PaperlessBilling),
  Contract = factor(Contract),
  InternetService = factor(InternetService),
  PaymentMethod = factor(PaymentMethod)
  
)


# standardize the quantitative variables (Kuhn, 2019)
churn_preproc <- preProcess(churn_df_init,
                          method = c("center", "scale", "nzv"))

churn_preproc

churn_df_transformed <- predict(churn_preproc, newdata = churn_df_init)

head(churn_df_transformed)

# Feature Selection  ----------------------------------------

# check for multicollinearity
# examine correlations among variables
cor_mat<-polycor::hetcor(as.data.frame(churn_df_transformed[,-1]), 
                         ML=T, use = "pairwise.complete.obs", std.err = F)
cor_mat <- as.matrix(cor_mat)

round(cor_mat,2)

# find and remove variable with large correlations (Kuhn, 2019)
highly_cor <- findCorrelation(cor_mat, cutoff = 0.8, names = TRUE)

highly_cor

# drop variable
churn_df_reduced <- subset(churn_df_transformed, select = -c(Bandwidth_GB_Year))

glimpse(churn_df_reduced)

# recursive feature selection (Kuhn, 2019)
set.seed(13468)
ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   verbose = TRUE)
rf_profile <- rfe(churn_df_reduced[,-c(1,2)], churn_df_reduced[[2]],
                  rfeControl = ctrl)
rf_profile

# get predictors 
predictors(rf_profile)

#plot

plot(rf_profile, type=c("o"))


# Data Preparation  ---------------------------------------

churn_df_knn <- churn_df_init %>% 
  select(Customer_id, Churn, Tenure, MonthlyCharge, Techie, 
         Multiple, StreamingTV, StreamingMovies, Contract, 
         InternetService)


glimpse(churn_df_knn)

# get dummy variables (Kuhn, 2019)
dummies <- dummyVars(~., data = churn_df_knn[,-c(1,2)])

head(predict(dummies, newdata = churn_df_knn[,-1]))

churn_knn_dummy <- cbind(churn_df_knn[,c(1:2)], 
                         predict(dummies, newdata = churn_df_knn)) 

head(churn_knn_dummy)



# Training-Test Split  ----------------------------------------
# (Kuhn, 2019)

# create split index 
set.seed(13468)
train_index <- createDataPartition(churn_knn_dummy$Churn, p = 0.8,
                                   list = FALSE,
                                   times = 1)

# create data sets
churn_train <- churn_knn_dummy[train_index, ]
churn_test <- churn_knn_dummy[-train_index, ]

# confirm splits
prop.table(table(churn_train$Churn))

prop.table(table(churn_test$Churn))

# standardize the quantitative variables (Kuhn, 2019) 
knn_preproc <- preProcess(churn_train[,3:4],
                             method = c("center", "scale"))

churn_train_final <- predict(knn_preproc, newdata = churn_train)
churn_test_final <- predict(knn_preproc, newdata = churn_test)

head(churn_train_final)

head(churn_test_final)



# KNN Analysis --  -----------------------------------------
# (Jawaharial, 2014; Kuhn, 2019; Kuhn, 2008)

# tune k values 
ctrls <- trainControl(method = "cv")

# specify K values to try 
knn_grid <- expand.grid(k = c(1,3,5,7,9,11,13,15,17,19,21,23))

# run cross validation model
knn_model <-train(Churn ~ .,
                  method = "knn",
                  data = churn_train_final[,-1],
                  trControl = ctrls, 
                  tuneGrid = knn_grid)

knn_model


# plot tuned models 
plot(knn_model)

knn_test_pred <- predict(knn_model, newdata = churn_test_final[,-c(1,2)])

# Accuracy on Test data 
confusionMatrix(knn_test_pred, churn_test_final$Churn, positive = "Yes")

# Get probabilities for test data 
knn_test_prob <- predict(knn_model, newdata = churn_test_final[,-c(1,2)], type = "prob")

# ROC 
knn_roc <- roc(churn_test_final$Churn, knn_test_prob[,2])

# AUC 
knn_roc

# ROC plot 
plot(knn_roc, print.auc = TRUE, xlim=c(1,0))



# Code References ---------------------------------------------------------

# Alboudkadel (2018, November 12). ggplot Axis Labels. Data Novia. https://www.datanovia.com/en/blog/ggplot-axis-labels/

# Batra, N., Spina, A., Blomquist, P. Campbell, F., Laurenson-Schafer, H., Florence, I., Fischer, N., Ndiaye, A., Coyer, L., Polonsky, J., Izawa, Y., Bailey, C., Molling, D., Berry, I., Buajitti, E., Mousset, M., Hollis, S., & Lin, W. (2021). The Epidemiologist R Handbook. doi: 10.5281/zenodo.4752646  

# Devlin, M. (2015, May 16). Tables and Plots of Counts and Proportions. https://rstudio-pubs-static.s3.amazonaws.com/80278_2a7b4c5561394b04ad3f135d89421919.html

# Jawaharlal, V. (2014, April 29). kNN Using caret R package.RPubs. https://rpubs.com/njvijay/16444

# Kassambara. (2017, January 09). ggplot2 – Easy Way to Mix Multiple Graphs on the Same Page. http://www.sthda.com/english/articles/24-ggpubr-publication-ready-plots/81-ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page/ 

# Kuhn, M. (2008). Building Predictive Models in R Using the caret Package. Journal of Statistical Software, 28(5), 1–26. https://doi.org/10.18637/jss.v028.i05

# Kuhn, M. (2019, March 27). The caret Package. https://topepo.github.io/caret/index.html

# rishabhchakrabortygfg, (2021). Change Color of Bars in Barchart using ggplot2 in R. GeeksforGeeks. https://www.geeksforgeeks.org/change-color-of-bars-in-barchart-using-ggplot2-in-r/#

# Sauer, S. (2016, Nov 3). How to plot a ‘percentage plot’ with ggplot2. https://data-se.netlify.app/2016/11/03/percentage_plot_ggplot2_v2/

# Tierney, N. (2023, February 2). Gallery of Missing Data Visualizations. CRAN. https://cran.r-project.org/web/packages/naniar/vignettes/naniar-visualisation.html

# Venkatachalam, S. (n.d.). SELECT VARIABLES (COLUMN) IN R USING DPLYR – SELECT () FUNCTION. Data Science Made Simple. Retrieved September 20, 2023, from https://www.datasciencemadesimple.com/select-variables-columns-r-using-dplyr-select-function/

# Wickham, H., Chang, W., Henry, L., Pedersen, T. L., Takahashi, K., Wilke, C., Woo, K., Yutani, H., Dunnington, D. (n.d.). Control aesthetic evaluation. Retrieved September 15, 2023 from https://ggplot2.tidyverse.org/reference/aes_eval.html 

# Zach. (2023, February 21). How to Round Values in Specific Columns Using dplyr. https://www.statology.org/dplyr-round/
  
