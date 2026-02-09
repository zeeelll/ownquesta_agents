#!/usr/bin/env python3
"""
Test script for Enhanced ML Assistant Agent with Goal Detection
Tests the new advanced validation and goal detection features
"""

import asyncio
import httpx
import json
import time
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
ML_ASSISTANT_URL = f"{BASE_URL}/ml-assistant"

# Test data scenarios
TEST_SCENARIOS = [
    {
        "name": "Customer Churn Classification",
        "goal": "I want to predict which customers are likely to churn in the next 3 months based on their usage patterns, demographics, and service history",
        "expected_task": "classification",
        "data": {
            "customer_id": range(1, 1001),
            "age": [25 + (i % 40) for i in range(1000)],
            "monthly_charges": [50 + (i % 100) for i in range(1000)],
            "tenure": [(i % 72) for i in range(1000)],
            "contract_type": ["Month-to-month", "One year", "Two year"][i % 3] for i in range(1000)],
            "total_charges": [50 * (i % 72) for i in range(1000)],
            "churn": [i % 10 < 2 for i in range(1000)]  # 20% churn rate
        }
    },
    {
        "name": "House Price Regression",  
        "goal": "Predict house prices based on location, size, age, and other property characteristics to help with real estate valuations",
        "expected_task": "regression",
        "data": {
            "property_id": range(1, 501),
            "sqft": [1000 + (i % 3000) for i in range(500)],
            "bedrooms": [(i % 5) + 1 for i in range(500)],
            "bathrooms": [(i % 3) + 1 for i in range(500)],
            "age": [(i % 50) for i in range(500)],
            "garage": [i % 2 for i in range(500)],  
            "price": [100000 + (i % 500000) for i in range(500)]
        }
    },
    {
        "name": "Customer Segmentation", 
        "goal": "Discover natural customer segments based on purchasing behavior, demographics, and engagement patterns for targeted marketing campaigns",
        "expected_task": "clustering",
        "data": {
            "customer_id": range(1, 301),
            "annual_spending": [1000 + (i % 10000) for i in range(300)],
            "frequency": [(i % 50) + 1 for i in range(300)],
            "recency": [(i % 365) for i in range(300)],
            "age": [18 + (i % 65) for i in range(300)],
            "gender": ["M" if i % 2 else "F" for i in range(300)],
            "engagement_score": [(i % 100) for i in range(300)]
        }
    },
    {
        "name": "Fraud Detection",
        "goal": "Detect fraudulent transactions in real-time to prevent financial losses and protect customers",
        "expected_task": "classification",  # Should detect binary classification
        "data": {
            "transaction_id": range(1, 1001),
            "amount": [10 + (i % 1000) for i in range(1000)],
            "merchant_category": [(i % 20) for i in range(1000)],
            "hour": [(i % 24) for i in range(1000)],
            "weekend": [i % 7 >= 5 for i in range(1000)],
            "card_type": ["credit", "debit"][i % 2] for i in range(1000)],
            "fraud": [i % 100 < 5 for i in range(1000)]  # 5% fraud rate
        }
    }
]


async def test_health_check():
    """Test basic health check"""
    logger.info("🔍 Testing health check...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{ML_ASSISTANT_URL}/health")
            assert response.status_code == 200
            data = response.json()
            
            logger.info(f"✅ Health check passed: {data['status']}")
            logger.info(f"   Agent: {data['agent']}")
            logger.info(f"   Version: {data['version']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False


async def test_goal_analysis(scenario):
    """Test goal analysis without dataset"""
    logger.info(f"🎯 Testing goal analysis for: {scenario['name']}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{ML_ASSISTANT_URL}/goal-analyze",
                data={"goal": scenario["goal"]}
            )
            
            if response.status_code != 200:
                logger.error(f"❌ Goal analysis failed with status {response.status_code}")
                return None
                
            data = response.json()
            goal_analysis = data["goal_analysis"]
            
            detected_task = goal_analysis["primary_task"] 
            confidence = goal_analysis["confidence"]
            expected_task = scenario["expected_task"]
            
            logger.info(f"   Detected task: {detected_task} (confidence: {confidence:.2%})")
            logger.info(f"   Expected task: {expected_task}")
            
            # Check if detection is reasonable (allowing some flexibility)
            if detected_task == expected_task or confidence < 0.6:
                logger.info("✅ Goal analysis reasonable")
            else:
                logger.warning(f"⚠️ Unexpected detection: expected {expected_task}, got {detected_task}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Goal analysis failed: {e}")
            return None


async def create_test_dataset(scenario):
    """Create test dataset for scenario"""
    df = pd.DataFrame(scenario["data"])
    
    # Create uploads directory if it doesn't exist  
    uploads_dir = Path("ml_assistant_agent/uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    filename = f"test_{scenario['name'].lower().replace(' ', '_')}.csv"
    filepath = uploads_dir / filename
    
    df.to_csv(filepath, index=False)
    logger.info(f"📁 Created test dataset: {filename} ({len(df)} rows, {len(df.columns)} columns)")
    
    return filename, filepath


async def test_advanced_validation(scenario):
    """Test advanced validation with goal detection"""
    logger.info(f"🚀 Testing advanced validation for: {scenario['name']}")
    
    # Create test dataset
    filename, filepath = await create_test_dataset(scenario)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Prepare file upload
            with open(filepath, "rb") as f:
                files = {"file": (filename, f, "text/csv")}
                data = {
                    "goal": scenario["goal"],
                    "target_column": None  # Let it auto-detect
                }
                
                response = await client.post(
                    f"{ML_ASSISTANT_URL}/advanced-validate",
                    files=files,
                    data=data
                )
            
            if response.status_code != 200:
                logger.error(f"❌ Advanced validation failed with status {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return None
            
            result = response.json()
            
            # Log key results
            task_type = result.get("task_type", "unknown")
            learning_type = result.get("learning_type", "unknown") 
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            logger.info(f"   🎯 Task Type: {task_type}")
            logger.info(f"   📚 Learning Type: {learning_type}")
            logger.info(f"   🎲 Confidence: {confidence:.2%}")
            logger.info(f"   💭 Reasoning: {reasoning}")
            
            # Log algorithm recommendations
            algorithms = result.get("algorithm_recommendations", [])
            if algorithms:
                logger.info("   🔧 Recommended Algorithms:")
                for alg in algorithms[:3]:  # Show top 3
                    priority = alg.get("priority", "unknown")
                    name = alg.get("name", "unknown")
                    logger.info(f"      • {name} ({priority} priority)")
            
            # Log data summary if available
            data_summary = result.get("data_summary", {})
            if data_summary:
                shape = data_summary.get("clean_shape", [0, 0])
                logger.info(f"   📊 Dataset: {shape[0]} rows × {shape[1]} columns")
                
                if "target_column" in data_summary:
                    logger.info(f"   🎯 Target Column: {data_summary['target_column']}")
            
            logger.info("✅ Advanced validation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Advanced validation failed: {e}")
            return None
        finally:
            # Cleanup test file
            try:
                filepath.unlink()
                logger.info(f"🗑️ Cleaned up test file: {filename}")
            except Exception:
                pass


async def run_comprehensive_test():
    """Run comprehensive test of enhanced ML assistant"""
    logger.info("🚀 Starting Enhanced ML Assistant Agent Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Health Check
    health_ok = await test_health_check()
    if not health_ok:
        logger.error("❌ Health check failed - aborting tests")
        return
    
    logger.info("")
    
    # Test 2: Goal Analysis for each scenario
    logger.info("🎯 Testing Goal Analysis")
    logger.info("-" * 40)
    
    for scenario in TEST_SCENARIOS:
        goal_result = await test_goal_analysis(scenario)
        if goal_result:
            logger.info("   Goal analysis successful\n")
        else:
            logger.warning("   Goal analysis had issues\n")
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Test 3: Advanced Validation for each scenario
    logger.info("🚀 Testing Advanced Validation")
    logger.info("-" * 40)
    
    for scenario in TEST_SCENARIOS:
        validation_result = await test_advanced_validation(scenario)
        if validation_result:
            logger.info("   Advanced validation successful\n")
        else:
            logger.warning("   Advanced validation had issues\n")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    logger.info("🏁 Test Suite Complete!")
    logger.info("=" * 60)


async def test_basic_endpoints():
    """Test basic endpoints to ensure they still work"""
    logger.info("🔧 Testing Basic Endpoints")
    logger.info("-" * 30)
    
    async with httpx.AsyncClient() as client:
        # Test main API health
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                logger.info("✅ Main API health check passed")
            else:
                logger.warning(f"⚠️ Main API health check returned {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Main API health check failed: {e}")
        
        # Test meta endpoint
        try:
            response = await client.get(f"{BASE_URL}/meta.json")
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Meta endpoint accessible")
                logger.info(f"   Available agents: {len(data.get('agents', []))}")
            else:
                logger.warning(f"⚠️ Meta endpoint returned {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Meta endpoint failed: {e}")


if __name__ == "__main__":
    print("Enhanced ML Assistant Agent Test Suite")
    print("=====================================")
    print()
    print("This script will test:")
    print("• Health check and basic connectivity")
    print("• Goal analysis for different ML scenarios")  
    print("• Advanced validation with auto task detection")
    print("• Algorithm recommendations")
    print()
    print("Make sure the OwnQuesta agents are running:")
    print("  cd ownquesta_agents")
    print("  python main.py")
    print()
    
    try:
        asyncio.run(test_basic_endpoints())
        print() 
        asyncio.run(run_comprehensive_test())
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        logging.exception("Full error details:")