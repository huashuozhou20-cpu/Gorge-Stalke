#!/usr/bin/env powershell
# 清理腾讯开悟训练任务产生的僵死容器和网络残余

Write-Host "开始清理腾讯开悟训练任务的容器和网络..." -ForegroundColor Green

# 停止并删除所有与 kaiwu 相关的容器
Write-Host "1. 停止并删除与 kaiwu 相关的容器..." -ForegroundColor Yellow
$containers = docker ps -a --filter "name=kaiwu" --format "{{.Names}}"
foreach ($container in $containers) {
    Write-Host "  停止容器: $container"
    docker stop $container -t 10 2>$null
    Write-Host "  删除容器: $container"
    docker rm $container -f 2>$null
}

# 特别处理 kaiwu-train-learner-1 容器
Write-Host "2. 特别处理 kaiwu-train-learner-1 容器..." -ForegroundColor Yellow
Write-Host "  停止容器: kaiwu-train-learner-1"
docker stop kaiwu-train-learner-1 -t 10 2>$null
Write-Host "  删除容器: kaiwu-train-learner-1"
docker rm kaiwu-train-learner-1 -f 2>$null

# 删除与该项目相关的网络
Write-Host "3. 删除与项目相关的网络..." -ForegroundColor Yellow
$networks = docker network ls --filter "name=gorge_chase" --format "{{.Name}}"
foreach ($network in $networks) {
    Write-Host "  删除网络: $network"
    docker network rm $network 2>$null
}

# 清理无标签的容器
Write-Host "4. 清理无标签的容器..." -ForegroundColor Yellow
docker container prune -f 2>$null

# 清理悬空的镜像
Write-Host "5. 清理悬空的镜像..." -ForegroundColor Yellow
docker image prune -f 2>$null

# 显示清理结果
Write-Host "6. 显示当前容器状态..." -ForegroundColor Yellow
docker ps -a --filter "name=kaiwu"

Write-Host "7. 显示当前网络状态..." -ForegroundColor Yellow
docker network ls --filter "name=gorge_chase"

Write-Host "清理完成！" -ForegroundColor Green
Write-Host "建议: 清理完成后，重新启动训练任务。" -ForegroundColor Cyan
