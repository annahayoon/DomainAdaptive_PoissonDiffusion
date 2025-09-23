#!/bin/bash
# Quick HPC cluster status checker

echo "🚀 HPC Cluster Status Check - $(date)"
echo "================================================"

echo ""
echo "📊 GPU Node Status:"
echo "-------------------"
sinfo -N -o '%N %T %G %m %c %O' --noheader | grep -i gpu

echo ""
echo "🎯 Available GPU Nodes:"
echo "----------------------"
sinfo -N -o '%N %T %G %m %c' --noheader | grep -i gpu | grep -i idle

echo ""
echo "🔒 Allocated GPU Nodes:"
echo "----------------------"
sinfo -N -o '%N %T %G %m %c' --noheader | grep -i gpu | grep -i alloc

echo ""
echo "📋 Current Jobs:"
echo "---------------"
squeue -o '%i %j %u %T %N %G %M %L' --noheader | head -20

echo ""
echo "👤 Your Jobs:"
echo "------------"
squeue -u $USER

echo ""
echo "🔧 Quick Commands:"
echo "-----------------"
echo "sinfo -N -o '%N %T %G %m %c' --noheader | grep -i gpu  # All GPU nodes"
echo "squeue -u \$USER                                        # Your jobs"
echo "scontrol show node <nodename>                           # Node details"
echo "squeue -o '%i %j %u %T %N %G %M %L' --noheader         # All jobs"
