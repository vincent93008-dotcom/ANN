# ============================================================================
# 類神經網路入門：從 Logistic Regression 到 ANN
# R 實作：(1) 神經元與反向傳播 (2) 二元分類 (3) 連續型迴歸
# 適用：統計學 / 機器學習教學，單一檔案可執行。
# 依賴：R 建議套件 nnet（若未安裝，先執行 install.packages("nnet")）。
# 執行：在 RStudio 中開啟並 Source，或 Rscript ANN_Logistic_to_Neural_Network.R
# 生成：ANN_R_Results/ 資料夾內的 CSV、TXT、PNG。
# 重要：資料均為模擬生成；程式在最後列印實際結果，勿先假設 ANN 一定最好。
# ============================================================================

options(stringsAsFactors = FALSE)
if (!requireNamespace("nnet", quietly = TRUE)) {
  stop("找不到 nnet 套件，請先執行：install.packages('nnet')")
}
set.seed(20261024)
out_dir <- file.path(getwd(), "ANN_R_Results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
report_path <- file.path(out_dir, "ANN_analysis_results.txt")
cat("ANN 教學實作報告\n", "執行日期：", as.character(Sys.Date()), "\n\n",
    file = report_path, sep = "")
append_report <- function(...) cat(..., file = report_path, append = TRUE, sep = "")

# ============================================================================
# 第一部分：手算一個 2-2-1 ANN 的 Forward/Backward Propagation
# ============================================================================
cat("=== 第一部分：一筆資料的前向/反向傳播 ===\n")
append_report("=== 第一部分：一筆資料的前向/反向傳播 ===\n")

sigmoid <- function(z) 1 / (1 + exp(-z))
x <- c(0.4, -0.2)  # 二個輸入變數
y <- 1             # 二元真實結果
W <- matrix(c(0.3, -0.1, 0.2, 0.4), nrow = 2, byrow = TRUE)
b_hidden <- c(0, 0.1)
v <- c(0.6, -0.7)
b_output <- 0.2
learning_rate <- 0.1

# (1) 前向傳播
z_hidden <- drop(W %*% x) + b_hidden
h <- sigmoid(z_hidden)
z_out <- sum(v * h) + b_output
p_before <- sigmoid(z_out)
loss_binary <- function(y, p) {
  p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  -y * log(p) - (1 - y) * log(1 - p)
}
loss_before <- loss_binary(y, p_before)

# (2) 反向傳播：Sigmoid + binary cross-entropy
# dL/dz_out = p - y
delta_out <- p_before - y
grad_v <- delta_out * h
grad_b_output <- delta_out
delta_hidden <- delta_out * v * h * (1 - h)
grad_W <- outer(delta_hidden, x)
grad_b_hidden <- delta_hidden

# (3) 使用一般梯度下降，更新所有權重與偏差項
W_new <- W - learning_rate * grad_W
b_hidden_new <- b_hidden - learning_rate * grad_b_hidden
v_new <- v - learning_rate * grad_v
b_output_new <- b_output - learning_rate * grad_b_output
h_new <- sigmoid(drop(W_new %*% x) + b_hidden_new)
p_after <- sigmoid(sum(v_new * h_new) + b_output_new)
loss_after <- loss_binary(y, p_after)

manual_summary <- data.frame(
  Stage = c("Before update", "After update"),
  Probability = c(p_before, p_after),
  BinaryCrossEntropy = c(loss_before, loss_after)
)
print(manual_summary, row.names = FALSE)
append_report(paste(capture.output(print(manual_summary, row.names = FALSE)),
                    collapse = "\n"), "\n\n")

# ============================================================================
# 第二部分：二元分類 — Logistic Regression vs. ANN
# 資料生成：真實 logit = 2.2 - 1.4*(x1^2+x2^2)
# 因為是非線性圓形決策邊界，單純線性 logit 模型的形式並不正確。
# 同時納入二次項 Logistic，公平呈現統計模型也可以處理非線性。
# ============================================================================
cat("\n=== 第二部分：二元分類 ===\n")
append_report("=== 第二部分：二元分類 ===\n")

n_class <- 1800
class_data <- data.frame(
  x1 = runif(n_class, -2, 2),
  x2 = runif(n_class, -2, 2)
)
class_data$true_prob <- plogis(2.2 - 1.4 *
                               (class_data$x1^2 + class_data$x2^2))
class_data$y <- rbinom(n_class, size = 1, prob = class_data$true_prob)
write.csv(class_data, file.path(out_dir, "classification_data.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# (1) 80% 訓練、20% 測試。三個方法使用完全相同的資料分割。
id_train <- sample.int(n_class, size = floor(0.8 * n_class))
train_c <- class_data[id_train, , drop = FALSE]
test_c <- class_data[-id_train, , drop = FALSE]

# (2) 只根據訓練資料的 X 計算標準化參數，避免資料洩漏。
xvars <- c("x1", "x2")
mu_c <- vapply(train_c[xvars], mean, numeric(1))
sd_c <- vapply(train_c[xvars], sd, numeric(1))
standardize <- function(d, variables, center, scale) {
  for (nm in variables) d[[nm]] <- (d[[nm]] - center[[nm]]) / scale[[nm]]
  d
}
train_c <- standardize(train_c, xvars, mu_c, sd_c)
test_c <- standardize(test_c, xvars, mu_c, sd_c)

# (3) 三種方法
# 模型 A：只含原始線性主效果的 Logistic Regression
fit_logistic <- glm(y ~ x1 + x2, family = binomial(), data = train_c)
# 模型 B：加入二次項的 Logistic Regression（較合適的模型形式）
fit_logistic_quad <- glm(y ~ x1 + x2 + I(x1^2) + I(x2^2),
                         family = binomial(), data = train_c)
# 模型 C：ANN，單一隱藏層（8 個神經元）、Sigmoid 輸出、交叉熵。
# nnet 內部使用 BFGS 最佳化，不是前段手算示範的基本梯度下降法。
fit_ann_class <- nnet::nnet(y ~ x1 + x2,
                            data = train_c, size = 8,
                            entropy = TRUE, decay = 0.002,
                            maxit = 700, trace = FALSE)

# (4) 測試集的預測機率
prob_logistic <- as.numeric(predict(fit_logistic, newdata = test_c,
                                     type = "response"))
prob_quad <- as.numeric(predict(fit_logistic_quad, newdata = test_c,
                                type = "response"))
prob_ann <- as.numeric(predict(fit_ann_class, newdata = test_c,
                               type = "raw"))

# (5) 評估：Accuracy、Sensitivity、Specificity、AUC、LogLoss
# AUC 以 Mann-Whitney / rank 公式計算，不需額外套件。
auc_rank <- function(truth, prob) {
  n1 <- sum(truth == 1)
  n0 <- sum(truth == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  (sum(rank(prob, ties.method = "average")[truth == 1]) -
     n1 * (n1 + 1) / 2) / (n1 * n0)
}
classification_metrics <- function(truth, prob, threshold = 0.5) {
  pred <- ifelse(prob >= threshold, 1, 0)
  TP <- sum(truth == 1 & pred == 1)
  TN <- sum(truth == 0 & pred == 0)
  FP <- sum(truth == 0 & pred == 1)
  FN <- sum(truth == 1 & pred == 0)
  eps <- 1e-12
  prob_safe <- pmin(pmax(prob, eps), 1 - eps)
  data.frame(
    Accuracy = mean(pred == truth),
    Sensitivity = if (TP + FN > 0) TP / (TP + FN) else NA_real_,
    Specificity = if (TN + FP > 0) TN / (TN + FP) else NA_real_,
    AUC = auc_rank(truth, prob),
    LogLoss = -mean(truth * log(prob_safe) +
                    (1 - truth) * log(1 - prob_safe))
  )
}
cm <- function(truth, prob) {
  table(Observed = factor(truth, levels = c(0, 1)),
        Predicted = factor(as.integer(prob >= 0.5), levels = c(0, 1)))
}
class_compare <- rbind(
  Linear_Logistic = classification_metrics(test_c$y, prob_logistic),
  Quadratic_Logistic = classification_metrics(test_c$y, prob_quad),
  ANN = classification_metrics(test_c$y, prob_ann)
)
print(round(class_compare, 4))
cat("\nANN Confusion Matrix:\n")
print(cm(test_c$y, prob_ann))
append_report(paste(capture.output(print(round(class_compare, 4))),
                    collapse = "\n"), "\n\n")
append_report("ANN Confusion Matrix:\n",
              paste(capture.output(print(cm(test_c$y, prob_ann))),
                    collapse = "\n"), "\n")
append_report("ANN convergence (0 means optimizer's convergence criterion met): ",
              fit_ann_class$convergence, "\n\n")
write.csv(cbind(Model = rownames(class_compare), class_compare),
          file.path(out_dir, "classification_metrics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# (6) 圖：三個模型的分類邊界（等預測機率 0.5）
gx <- seq(min(c(train_c$x1, test_c$x1)),
          max(c(train_c$x1, test_c$x1)), length.out = 110)
gy <- seq(min(c(train_c$x2, test_c$x2)),
          max(c(train_c$x2, test_c$x2)), length.out = 110)
grid <- expand.grid(x1 = gx, x2 = gy)
grid_p <- list(
  as.numeric(predict(fit_logistic, grid, type = "response")),
  as.numeric(predict(fit_logistic_quad, grid, type = "response")),
  as.numeric(predict(fit_ann_class, grid, type = "raw"))
)
figure_file <- file.path(out_dir, "classification_boundaries.png")
png(figure_file, width = 1500, height = 540, res = 130)
par(mfrow = c(1, 3), mar = c(4.2, 4.1, 3.3, 1.0))
model_names <- c("Linear logistic", "Quadratic logistic", "ANN")
for (m in seq_along(grid_p)) {
  plot(test_c$x1, test_c$x2,
       pch = 16, cex = 0.60,
       col = ifelse(test_c$y == 1, "#D45C51", "#3387B8"),
       xlab = "Standardized x1", ylab = "Standardized x2",
       main = model_names[m], xlim = range(gx), ylim = range(gy))
  z <- matrix(grid_p[[m]], nrow = length(gx), ncol = length(gy))
  if (min(z) <= 0.5 && max(z) >= 0.5) {
    contour(gx, gy, z, levels = 0.5, add = TRUE,
            drawlabels = FALSE, lwd = 2.4, col = "#161616")
  }
}
dev.off()

# ============================================================================
# 第三部分：連續型迴歸 — Linear Regression vs. ANN
# 資料生成：y = sin(1.7*x1) + 0.5*x2^2 + 0.3*x1*x2 + noise
# 同時加入正確非線性基底的線性模型，比較 ANN 和有結構的統計模型。
# ============================================================================
cat("\n=== 第三部分：連續型迴歸 ===\n")
append_report("=== 第三部分：連續型迴歸 ===\n")

n_reg <- 1800
reg_data <- data.frame(
  x1 = runif(n_reg, -3, 3),
  x2 = runif(n_reg, -2, 2)
)
reg_data$y <- sin(1.7 * reg_data$x1) +
  0.5 * reg_data$x2^2 +
  0.3 * reg_data$x1 * reg_data$x2 +
  rnorm(n_reg, mean = 0, sd = 0.25)
write.csv(reg_data, file.path(out_dir, "regression_data.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

id_train_r <- sample.int(n_reg, size = floor(0.8 * n_reg))
train_r <- reg_data[id_train_r, , drop = FALSE]
test_r <- reg_data[-id_train_r, , drop = FALSE]

# (1) 傳統線性迴歸與具有合適非線性基底的迴歸
fit_lm <- lm(y ~ x1 + x2, data = train_r)
fit_lm_basis <- lm(y ~ sin(1.7 * x1) + I(x2^2) + I(x1 * x2),
                   data = train_r)
pred_lm <- as.numeric(predict(fit_lm, newdata = test_r))
pred_lm_basis <- as.numeric(predict(fit_lm_basis, newdata = test_r))

# (2) ANN：標準化 X 和 Y（全部只使用訓練集統計量）
mu_r <- vapply(train_r[xvars], mean, numeric(1))
sd_r <- vapply(train_r[xvars], sd, numeric(1))
mu_y <- mean(train_r$y)
sd_y <- sd(train_r$y)
train_r_nn <- data.frame(
  x1 = (train_r$x1 - mu_r[["x1"]]) / sd_r[["x1"]],
  x2 = (train_r$x2 - mu_r[["x2"]]) / sd_r[["x2"]],
  y = (train_r$y - mu_y) / sd_y
)
test_r_nn <- data.frame(
  x1 = (test_r$x1 - mu_r[["x1"]]) / sd_r[["x1"]],
  x2 = (test_r$x2 - mu_r[["x2"]]) / sd_r[["x2"]]
)
# linout=TRUE：輸出神經元使用 identity/linear activation，非 Sigmoid。
fit_ann_reg <- nnet::nnet(y ~ x1 + x2,
                          data = train_r_nn,
                          size = 10, linout = TRUE,
                          decay = 0.002, maxit = 850,
                          trace = FALSE)
# 反標準化，回復到 Y 的原始量尺。
pred_ann_reg_z <- as.numeric(predict(fit_ann_reg,
                                      newdata = test_r_nn, type = "raw"))
pred_ann_reg <- pred_ann_reg_z * sd_y + mu_y

# (3) 測試集評估：RMSE / MAE / R-squared
regression_metrics <- function(truth, pred) {
  mse <- mean((truth - pred)^2)
  data.frame(
    RMSE = sqrt(mse),
    MAE = mean(abs(truth - pred)),
    Test_R2 = 1 - sum((truth - pred)^2) /
      sum((truth - mean(truth))^2)
  )
}
reg_compare <- rbind(
  Linear_Regression = regression_metrics(test_r$y, pred_lm),
  Basis_Regression = regression_metrics(test_r$y, pred_lm_basis),
  ANN = regression_metrics(test_r$y, pred_ann_reg)
)
print(round(reg_compare, 4))
append_report(paste(capture.output(print(round(reg_compare, 4))),
                    collapse = "\n"), "\n\n")
append_report("ANN regression convergence (0 means optimizer's convergence criterion met): ",
              fit_ann_reg$convergence, "\n\n")
write.csv(cbind(Model = rownames(reg_compare), reg_compare),
          file.path(out_dir, "regression_metrics.csv"),
          row.names = FALSE, fileEncoding = "UTF-8")

# (4) 圖：真實 Y 與預測 Y；越靠近 45 度線，預測越準確。
all_pred <- list(pred_lm, pred_lm_basis, pred_ann_reg)
lims <- range(c(test_r$y, unlist(all_pred)))
png(file.path(out_dir, "regression_predictions.png"),
    width = 1500, height = 540, res = 130)
par(mfrow = c(1, 3), mar = c(4.2, 4.2, 3.2, 1.0))
for (m in seq_along(all_pred)) {
  plot(test_r$y, all_pred[[m]],
       pch = 16, cex = 0.6, col = "#3387B899",
       xlim = lims, ylim = lims,
       xlab = "Observed Y", ylab = "Predicted Y",
       main = c("Linear regression", "Basis regression", "ANN")[m])
  abline(a = 0, b = 1, col = "#CF5548", lwd = 2)
}
dev.off()

# ============================================================================
# 第四部分：保存可供學生延伸分析的模型與輸出
# ============================================================================
saveRDS(
  list(classification = list(logistic = fit_logistic,
                             logistic_quadratic = fit_logistic_quad,
                             ann = fit_ann_class),
       regression = list(linear = fit_lm,
                         basis = fit_lm_basis,
                         ann = fit_ann_reg),
       classification_metrics = class_compare,
       regression_metrics = reg_compare),
  file = file.path(out_dir, "ANN_fitted_models.rds")
)
append_report("執行套件 nnet 版本：", as.character(utils::packageVersion("nnet")),
              "\n輸出資料夾：", normalizePath(out_dir, winslash = "/"), "\n")
cat("\n全部完成。輸出結果位置：\n", normalizePath(out_dir, winslash = "/"), "\n")
cat("請比較線性 Logistic、二次項 Logistic 與 ANN；\n")
cat("再比較一般線性迴歸、正確非線性基底迴歸與 ANN。\n")
