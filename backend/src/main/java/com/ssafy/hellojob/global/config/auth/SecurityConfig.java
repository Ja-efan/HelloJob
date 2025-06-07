package com.ssafy.hellojob.global.config.auth;

import com.ssafy.hellojob.global.auth.handler.CustomAccessDeniedHandler;
import com.ssafy.hellojob.global.auth.handler.CustomAuthenticationEntryPoint;
import com.ssafy.hellojob.global.auth.service.OAuth2UserService;
import com.ssafy.hellojob.global.auth.service.handler.OAuth2FailureHandler;
import com.ssafy.hellojob.global.auth.service.handler.OAuth2SuccessHandler;
import com.ssafy.hellojob.global.auth.token.JwtAuthenticationFilter;
import com.ssafy.hellojob.global.filter.SseExceptionFilter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.Arrays;

@Slf4j
@Configuration
@EnableWebSecurity
@RequiredArgsConstructor
public class SecurityConfig {

    private final OAuth2UserService oAuth2UserService;
    private final OAuth2SuccessHandler oAuth2SuccessHandler;
    private final OAuth2FailureHandler oAuth2FailureHandler;
    private final JwtAuthenticationFilter jwtAuthenticationFilter;
    private final CustomAuthenticationEntryPoint customAuthenticationEntryPoint;
    private final CustomAccessDeniedHandler customAccessDeniedHandler;
    private final SseExceptionFilter sseExceptionFilter;

    @Value("${frontend.url}")
    private String frontendUrl;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                .csrf(csrf -> csrf.disable())
                .cors(cors -> cors.configurationSource(corsConfigurationSource()))
                //세션 비활성화
                .sessionManagement(session ->
                        session.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                )
                //요청 권한 설정
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/api/v1/auth/**", "/oauth2/**", "/login/**", "/error").permitAll()
                        .requestMatchers("/sse/**").authenticated()
                        .anyRequest().authenticated()
                )
                .oauth2Login(oauth2 -> oauth2
                        .authorizationEndpoint(endpoint -> endpoint
                                .baseUri("/oauth2/authorization")
                        )
                        .userInfoEndpoint(userInfo ->
                                userInfo.userService(oAuth2UserService)
                        )
                        .successHandler(oAuth2SuccessHandler)
                        .failureHandler(oAuth2FailureHandler)
                )
                //JWT 필터가 UsernamePasswordAuthenticationFilter 전에 실행되도록 지정, 비번 검증 전에 토큰의 유효성 검증
                .addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class)
                .addFilterAfter(sseExceptionFilter, JwtAuthenticationFilter.class)
                // 인증 실패 핸들러 설정
                .exceptionHandling(handling -> handling
                        // JWT 인증 실패 시 401 반환
                        .authenticationEntryPoint(customAuthenticationEntryPoint)
                        // 403 처리
                        .accessDeniedHandler(customAccessDeniedHandler)
                );
        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOrigins(Arrays.asList(frontendUrl, "http://localhost:5173"));
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        // 쿠키 및 Authorization 헤더 노출 허용
        configuration.setExposedHeaders(Arrays.asList("Authorization"));

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }

}