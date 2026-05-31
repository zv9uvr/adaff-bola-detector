import { Injectable, NestInterceptor, ExecutionContext, CallHandler, ForbiddenException } from '@nestjs/common';
import { Observable } from 'rxjs';
import * as fs from 'fs';
import axios from 'axios';

@Injectable()
export class BolaDefenseInterceptor implements NestInterceptor {
	private criticalPaths: any[];

	constructor() {
		// 1. LLM이 생성한 '고위험 경로(타겟 방어 뇌)' 로드
		try {
			const data = fs.readFileSync('./bola_critical_paths.json', 'utf8');
			this.criticalPaths = JSON.parse(data).critical_paths;
			console.log('🛡️ BOLA 방어 인터셉터 가동: LLM 룰 로드 완료');
		} catch (e) {
			this.criticalPaths = [];
		}
	}

	async intercept(context: ExecutionContext, next: CallHandler): Promise<Observable<any>> {
		const req = context.switchToHttp().getRequest();
		const currentPath = req.route ? req.route.path : req.path;

		// 2. 현재 경로가 LLM이 지목한 고위험 경로인지 검사
		const isCritical = this.criticalPaths.some(p => {
			const basePath = p.path.split('{')[0];
			return currentPath.includes(basePath);
		});

		if (isCritical) {
			// 3. 고위험 경로라면, 사용자의 최근 20개 트래픽 피처(v8은 19개 피처) 추출
			const userFeatures = req.session?.recentFeatures || []; 
			
			// 윈도우 사이즈(20)가 충족되었을 때만 검사
			if (userFeatures.length === 20) {
				try {
					// 4. Python AI 서버(v8)에 판별 요청
					const aiResponse = await axios.post('http://localhost:8000/predict', {
						features: userFeatures
					});

					// 5. AI(Transformer)가 공격이라고 확신하면 403 에러 발생!
					if (aiResponse.data.is_attack) {
						console.warn(`🚨 [차단됨] BOLA 공격 감지 | 유저 IP: ${req.ip} | 확률: ${(aiResponse.data.score * 100).toFixed(1)}%`);
						throw new ForbiddenException('비정상적인 타인 객체 접근(BOLA)이 감지되어 연결이 차단되었습니다.');
					}
				} catch (error) {
					if (error instanceof ForbiddenException) throw error;
					console.error('AI 서버 통신 에러:', error.message);
				}
			}
		}

		return next.handle();
	}
}