package com.ssafy.hellojob.global.auth.service;

import com.ssafy.hellojob.domain.user.entity.Provider;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.repository.UserRepository;
import com.ssafy.hellojob.global.auth.entity.Auth;
import com.ssafy.hellojob.global.auth.repository.AuthRepository;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import com.ssafy.hellojob.global.util.AESUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.oauth2.client.userinfo.DefaultOAuth2UserService;
import org.springframework.security.oauth2.client.userinfo.OAuth2UserRequest;
import org.springframework.security.oauth2.core.OAuth2AuthenticationException;
import org.springframework.security.oauth2.core.OAuth2Error;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

@Slf4j
@Service
@Transactional
@RequiredArgsConstructor
public class OAuth2UserService extends DefaultOAuth2UserService {

    private final UserRepository userRepository;
    private final AuthRepository authRepository;
    private final AESUtil aesUtil;


    @Override
    public OAuth2User loadUser(OAuth2UserRequest userRequest) throws OAuth2AuthenticationException {
        OAuth2User oauth2User = super.loadUser(userRequest);

        String provider = userRequest.getClientRegistration().getRegistrationId();  // "google"
        String providerId = oauth2User.getName();  // 구글에서 제공하는 고유 ID (sub)

        Map<String, Object> attributes = oauth2User.getAttributes();

        // 구글은 최상위 레벨에 사용자 정보를 제공함
        String email = (String) attributes.get("email");
        String nickname = (String) attributes.get("name");

        User user = userRepository.findByEmail(email)
                .orElseGet(() -> createUser(email, nickname, provider, providerId));

        // 발표 시연 시 팀원 제외 다른 사람들 로그인 차단
//        if(user.getUserId() > 6) {
//            throw new BaseException(INVALID_USER);
//        }

        // 사용자가 있고 탈퇴 상태라면 예외 발생
        if (Boolean.TRUE.equals(user.getWithdraw())) {
            throw new OAuth2AuthenticationException(
                    new OAuth2Error("user_withdrawn", "User has withdrawn from the service", null)
            );
        }

        findOrCreateAuth(user);
        return new UserPrincipal(oauth2User, user);
    }

    private User createUser(String email, String nickname, String provider, String providerId) {
        String encryptProviderId = aesUtil.encrypt(providerId);
        User user = User.builder()
                .email(email)
                .nickname(nickname)
                .provider(provider.equals("google") ? Provider.GOOGLE : null)
                .providerId(encryptProviderId)
                .build();
        return userRepository.save(user);
    }

    private void findOrCreateAuth(User user) {
        if (!authRepository.existsByUser(user)) {
            createAuth(user);
        }
    }

    private void createAuth(User user) {
        Auth auth = Auth.builder()
                .user(user)
                .build();
        authRepository.save(auth);
    }

}

