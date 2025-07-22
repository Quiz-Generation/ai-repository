pipeline {
    agent any
    
    environment {
        IMAGE_TAG = "${new Date().format('yy.MM.dd')}-${BUILD_NUMBER}"
        FINAL_IMAGE = "ghcr.io/quiz-generation/ai-repository:${IMAGE_TAG}"
        GHCR_CREDENTIALS = credentials('ghcr-credentials')
    }
    
    stages {
        stage('환경 정보') {
            steps {
                echo '   빌드 정보:'
                echo "   브랜치: ${env.BRANCH_NAME}"
                echo "   빌드 번호: ${BUILD_NUMBER}"
                echo "   이미지 태그: ${IMAGE_TAG}"
                echo "   최종 이미지: ${FINAL_IMAGE}"
                echo "   GitOps: ArgoCD가 자동 배포 처리"
                
                sh '''
                    echo "Docker 버전:"
                    docker --version
                    echo "현재 워크스페이스:"
                    pwd
                    ls -la
                '''
            }
        }
        
        stage('빌드 파일 확인') {
            steps {
                echo '빌드에 필요한 파일들 확인...'
                sh '''
                    echo "소스 코드 확인:"
                    ls -la src/app/ | head -5
                    
                    echo "빌드 파일 확인:"
                    ls -la | grep -E "(Dockerfile|requirements.txt|gunicorn|setup.py)"
                    
                    echo "requirements.txt 내용 확인:"
                    head -10 requirements.txt
                    
                    echo "gunicorn 설정 파일 확인:"
                    ls -la gunicorn*
                '''
            }
        }
        
        stage('Docker 이미지 빌드') {
            steps {
                echo 'Docker 이미지 빌드 시작...'
                script {
                    docker.build("${FINAL_IMAGE}")
                    echo "✅ 이미지 빌드 완료: ${IMAGE_TAG}"
                }
            }
        }
        
        stage('이미지 테스트') {
            steps {
                echo '컨테이너 실행 테스트...'
                sh """
                    docker images | grep ai-repository
                    echo "컨테이너 실행 테스트..."
                    docker run --rm --name test-${BUILD_NUMBER} -d -p \$((8000 + ${BUILD_NUMBER})):8000 -e ENVIRONMENT=test ${FINAL_IMAGE}
                    
                    echo "컨테이너 시작 대기..."
                    sleep 10
                    
                    echo "컨테이너 상태 확인..."
                    docker ps | grep test-${BUILD_NUMBER} || echo "컨테이너가 실행되지 않음"
                    
                    echo "컨테이너 로그 확인..."
                    docker logs test-${BUILD_NUMBER} || true
                    
                    echo "테스트 컨테이너 정지..."
                    docker stop test-${BUILD_NUMBER} || true
                """
            }
        }
        
        stage('GHCR에 이미지 푸시') {
            steps {
                echo 'GitHub Container Registry에 이미지 푸시...'
                script {
                    docker.withRegistry('https://ghcr.io', 'ghcr-credentials') {
                        docker.image("${FINAL_IMAGE}").push()
                        docker.image("${FINAL_IMAGE}").push("latest")
                        echo "✅ 이미지 푸시 완료: ${FINAL_IMAGE}"
                    }
                }
            }
        }
        
        stage('ArgoCD 자동 배포 대기') {
            steps {
                echo 'ArgoCD가 새 이미지를 감지하여 자동 배포를 진행합니다...'
                echo "배포될 이미지: ${FINAL_IMAGE}"
                echo "ArgoCD 대시보드에서 배포 상태를 확인하세요."
            }
        }
        
        stage('리소스 정리') {
            steps {
                echo '불필요한 Docker 리소스 정리...'
                sh '''
                    docker stop test-${BUILD_NUMBER} || true
                    docker rm test-${BUILD_NUMBER} || true
                    docker system prune -f
                '''
            }
        }
    }
    
    post {
        success {
            echo '✅ CI/CD 파이프라인 성공!'
            echo "이미지: ${FINAL_IMAGE}"
            echo 'ArgoCD에서 배포 상태를 확인하세요.'
        }
        failure {
            echo '❌ CI/CD 파이프라인 실패!'
            echo 'Jenkins 로그를 확인하여 문제를 해결하세요.'
            sh '''
                docker stop test-${BUILD_NUMBER} || true
                docker rm test-${BUILD_NUMBER} || true
                docker system prune -f
            '''
        }
    }
}
