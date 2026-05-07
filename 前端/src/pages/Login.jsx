import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Form, Input, Button, Tabs, Modal, message } from 'antd'
import { UserOutlined, LockOutlined, TeamOutlined, SafetyCertificateOutlined } from '@ant-design/icons'
import { authApi, getApiErrorMessage } from '../api'

export default function Login() {
  const [loading, setLoading] = useState(false)
  const [registerModal, setRegisterModal] = useState(false)
  const [registerLoading, setRegisterLoading] = useState(false)
  const [captcha, setCaptcha] = useState('1234')
  const [loginForm] = Form.useForm()
  const [registerForm] = Form.useForm()
  const navigate = useNavigate()

  const onLogin = async (v) => {
    if (v.captcha?.toLowerCase() !== captcha.toLowerCase()) {
      message.error('验证码错误')
      return
    }
    setLoading(true)
    try {
      const res = await authApi.login(v.username, v.password)
      localStorage.setItem('token', res.data.access_token)
      localStorage.setItem('user', JSON.stringify(res.data.user))
      message.success('登录成功')
      navigate('/')
    } catch (e) {
      const d = e.response?.data?.detail
      message.error(typeof d === 'string' ? d : getApiErrorMessage(e))
    } finally {
      setLoading(false)
    }
  }

  const onRegister = async () => {
    try {
      const v = await registerForm.validateFields()
      setRegisterLoading(true)
      await authApi.register({ username: v.username, password: v.password, group_id: v.group_id || undefined })
      message.success('注册成功，请登录')
      setRegisterModal(false)
      registerForm.resetFields()
      loginForm.setFieldValue('username', v.username)
    } catch (e) {
      if (e.errorFields) return
      const d = e.response?.data?.detail
      message.error(typeof d === 'string' ? d : getApiErrorMessage(e))
    } finally {
      setRegisterLoading(false)
    }
  }

  const LoginForm = () => (
    <Form form={loginForm} onFinish={onLogin} size="large" className="glass-input">
      <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
        <Input prefix={<UserOutlined style={{ color: 'rgba(255,255,255,0.5)' }} />} placeholder="用户名" />
      </Form.Item>
      <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
        <Input.Password prefix={<LockOutlined style={{ color: 'rgba(255,255,255,0.5)' }} />} placeholder="密码" />
      </Form.Item>
      <Form.Item name="captcha" rules={[{ required: true, message: '请输入验证码' }]}>
        <Input
          prefix={<SafetyCertificateOutlined style={{ color: 'rgba(255,255,255,0.5)' }} />}
          placeholder="验证码"
          addonAfter={
            <span style={{ cursor: 'pointer', color: '#40a9ff', userSelect: 'none' }} onClick={() => setCaptcha(String(Math.random()).slice(2, 6))}>
              {captcha}
            </span>
          }
        />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit" block loading={loading} className="tech-btn" style={{ height: 44 }}>
          登录
        </Button>
      </Form.Item>
    </Form>
  )

  return (
    <div className="main-bg" style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div
        className="glass-card"
        style={{
          width: 440,
          padding: 32,
          background: 'rgba(255,255,255,0.06)',
          backdropFilter: 'blur(24px)',
          border: '1px solid rgba(255,255,255,0.15)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.3), 0 0 40px rgba(24,144,255,0.1)',
        }}
      >
        <h1 style={{ textAlign: 'center', color: '#e8f4ff', marginBottom: 24, fontSize: 22, fontWeight: 600 }}>
          短期负荷预测管理系统
        </h1>
        <Tabs
          items={[
            {
              key: 'user',
              label: <span><UserOutlined /> 用户登录</span>,
              children: (
                <>
                  <LoginForm />
                  <div style={{ textAlign: 'center', marginTop: 16 }}>
                    <Button type="link" style={{ color: '#40a9ff' }} onClick={() => setRegisterModal(true)}>
                      没有账号？立即注册
                    </Button>
                  </div>
                </>
              ),
            },
            {
              key: 'admin',
              label: <span><TeamOutlined /> 管理员登录</span>,
              children: (
                <>
                  <LoginForm />
                  <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, textAlign: 'center', marginTop: 16 }}>
                    管理员账号由总管理员创建
                  </div>
                </>
              ),
            },
          ]}
          style={{ color: '#fff' }}
        />
      </div>

      <Modal
        title="用户注册"
        open={registerModal}
        onOk={onRegister}
        onCancel={() => { setRegisterModal(false); registerForm.resetFields() }}
        confirmLoading={registerLoading}
        okText="注册"
      >
        <Form form={registerForm} layout="vertical">
          <Form.Item name="username" label="用户名" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true, min: 6 }]}>
            <Input.Password />
          </Form.Item>
          <Form.Item name="group_id" label="小组ID（选填）">
            <Input placeholder="若由管理员分配小组可填写" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
